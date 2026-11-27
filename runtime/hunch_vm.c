/* Hunch virtual machine and learning engine.
 *
 * One file, no dependencies beyond libc and libm.
 *
 * The whole point: every world's state is a flat POD block (values + PRNG +
 * step counter). Snapshotting the universe is therefore a single memcpy, which
 * makes exact counterfactual branching -- "run this same instant again, but
 * take a different action" -- essentially free. That is what makes
 * rewind-and-fork reinforcement learning possible as a language primitive.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Limits (must match hunch/codegen.py)                                */
/* ------------------------------------------------------------------ */

#define MAX_SLOTS   512
#define MAX_OBS     64
#define MAX_ACT     32
#define MAX_AGENTS  8
#define MAX_LISTED  65536.0     /* outcomes of one randint the exact graph will follow */
#define MAX_FUNCS   32
#define MAX_DECLS   96
#define MAX_SCRATCH 64
#define STACK_SIZE  1024
#define CALL_DEPTH  64
#define HID         32
#define INSN_BUDGET 400000
#define MAX_DEC_PER_STEP 8

/* Every run is scored on this same stream of episodes, so two algorithms are
 * compared on identical draws rather than on luck. */
#define EVAL_SEED 987654321ULL

enum {
    OP_HALT = 0, OP_PUSH_CONST, OP_PUSH_VAR, OP_STORE_VAR, OP_PUSH_IDX,
    OP_STORE_IDX, OP_POP, OP_DUP, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_NEG, OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE, OP_AND, OP_OR, OP_NOT,
    OP_JUMP, OP_JUMP_FALSE, OP_CALL, OP_RET, OP_SCRATCH, OP_COPY_VARS,
    OP_HUNCH, OP_REWARD, OP_DONE, OP_RAND, OP_RANDINT, OP_ABS, OP_MIN, OP_MAX,
    OP_CLAMP, OP_FLOOR, OP_SQRT, OP_EXP, OP_STEP_COUNT, OP_ASSERT, OP_LEGAL,
    OP_ZERO
};

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */

/* Something that stops the run: a broken rule of the world, an action that
 * cannot be played, a file that cannot be read. As a program this prints the
 * reason and exits with `code`. As a library (see the end of this file) the
 * caller has set a trap, and the reason goes back to it instead. */
static __thread jmp_buf g_trap;
static __thread int g_trap_on = 0;
static __thread char g_trap_msg[320];
/* Set while a search is trying continuations: a rule broken in one of them
 * was not broken by the episode itself, and the message says so. */
static __thread int g_searching = 0;

static void fatal(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_trap_msg, sizeof(g_trap_msg), fmt, ap);
    va_end(ap);
    g_searching = 0;
    if (g_trap_on) longjmp(g_trap, code);
    fprintf(stderr, "%s\n", g_trap_msg);
    exit(code);
}

static void die(const char *msg) {
    fatal(1, "hunch-vm: %s", msg);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static void rng_seed(uint64_t s[4], uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B97F4A7C15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        s[i] = x ^ (x >> 31);
    }
    if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 1;
}

static inline uint64_t rng_next(uint64_t s[4]) {
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

static inline double rng_double(uint64_t s[4]) {
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* A draw from Gamma(alpha, 1), for Dirichlet noise (Marsaglia and Tsang). */
static double rng_gamma(uint64_t s[4], double alpha) {
    double boost = 1.0;
    if (alpha < 1.0) {
        double u = rng_double(s);
        boost = pow(u > 1e-300 ? u : 1e-300, 1.0 / alpha);
        alpha += 1.0;
    }
    double d = alpha - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            double u1 = rng_double(s), u2 = rng_double(s);
            x = sqrt(-2.0 * log(u1 > 1e-300 ? u1 : 1e-300)) * cos(6.283185307179586 * u2);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        double u = rng_double(s);
        if (u < 1.0 - 0.0331 * x * x * x * x ||
            log(u > 1e-300 ? u : 1e-300) < 0.5 * x * x + d * (1.0 - v + log(v)))
            return d * v * boost;
    }
}

/* ------------------------------------------------------------------ */
/* Program                                                             */
/* ------------------------------------------------------------------ */

typedef struct { char name[48]; int obs_dim; int n_act; } AgentInfo;
typedef struct { char name[48]; int base; int len; int is_int; } DeclInfo;
typedef struct { char name[48]; int entry; int frame_base; int nparams; } FuncInfo;

typedef struct {
    int nslots, nconsts, ncode, nagents, nfuncs, horizon, ndecls;
    int init_entry, step_entry;
    int nstate;         /* slots [0, nstate) are the `state` block; the rest are locals */
    char world[64];
    char path[256];
    AgentInfo agents[MAX_AGENTS];
    DeclInfo decls[MAX_DECLS];
    FuncInfo funcs[MAX_FUNCS];
    double *consts;
    int32_t *code;
    double *defaults;
} Program;

static void read_exact(FILE *f, void *buf, size_t n) {
    if (fread(buf, 1, n, f) != n) die("truncated bytecode file");
}

static uint32_t read_u32(FILE *f) {
    uint32_t v;
    read_exact(f, &v, 4);
    return v;
}

static void read_str(FILE *f, char *out, size_t cap) {
    uint32_t n = read_u32(f);
    if (n >= cap) die("name too long in bytecode");
    read_exact(f, out, n);
    out[n] = 0;
}

static void load_program(const char *path, Program *p) {
    FILE *f = fopen(path, "rb");
    if (!f) fatal(1, "hunch-vm: cannot open %s: %s", path, strerror(errno));
    memset(p, 0, sizeof(*p));
    char magic[8];
    read_exact(f, magic, 8);
    if (memcmp(magic, "HUNCHBC\x01", 8) != 0) die("not a hunch bytecode file");
    if (read_u32(f) != 1) die("unsupported bytecode version");

    uint32_t nslots = read_u32(f), nconsts = read_u32(f), ncode = read_u32(f);
    uint32_t nagents = read_u32(f), nfuncs = read_u32(f), horizon = read_u32(f);
    uint32_t ndecls = read_u32(f);
    uint32_t init_entry = read_u32(f), step_entry = read_u32(f);

    if (nslots > MAX_SLOTS) die("program needs too many state slots");
    if (nagents > MAX_AGENTS) die("too many hunch agents");
    if (nfuncs > MAX_FUNCS) die("too many functions");
    if (ndecls > MAX_DECLS) die("too many state declarations");

    p->nslots = (int)nslots;
    p->nconsts = (int)nconsts;
    p->ncode = (int)ncode;
    p->nagents = (int)nagents;
    p->nfuncs = (int)nfuncs;
    p->horizon = (int)horizon;
    p->ndecls = (int)ndecls;
    p->init_entry = (int)init_entry;
    p->step_entry = (int)step_entry;

    read_str(f, p->world, sizeof(p->world));
    read_str(f, p->path, sizeof(p->path));

    for (int i = 0; i < p->nagents; i++) {
        read_str(f, p->agents[i].name, sizeof(p->agents[i].name));
        p->agents[i].obs_dim = (int)read_u32(f);
        p->agents[i].n_act = (int)read_u32(f);
    }
    for (int i = 0; i < p->ndecls; i++) {
        read_str(f, p->decls[i].name, sizeof(p->decls[i].name));
        p->decls[i].base = (int)read_u32(f);
        p->decls[i].len = (int)read_u32(f);
        p->decls[i].is_int = (int)read_u32(f);
        if (p->decls[i].base + p->decls[i].len > p->nstate)
            p->nstate = p->decls[i].base + p->decls[i].len;
    }
    for (int i = 0; i < p->nfuncs; i++) {
        read_str(f, p->funcs[i].name, sizeof(p->funcs[i].name));
        p->funcs[i].entry = (int)read_u32(f);
        p->funcs[i].frame_base = (int)read_u32(f);
        p->funcs[i].nparams = (int)read_u32(f);
    }

    p->consts = (double *)xmalloc(sizeof(double) * (size_t)(nconsts ? nconsts : 1));
    if (nconsts) read_exact(f, p->consts, sizeof(double) * nconsts);

    p->code = (int32_t *)xmalloc(sizeof(int32_t) * 4 * (size_t)(ncode ? ncode : 1));
    if (ncode) read_exact(f, p->code, sizeof(int32_t) * 4 * ncode);

    p->defaults = (double *)xmalloc(sizeof(double) * (size_t)(nslots ? nslots : 1));
    if (nslots) read_exact(f, p->defaults, sizeof(double) * nslots);

    fclose(f);
}

/* ------------------------------------------------------------------ */
/* State -- one flat block; snapshotting is memcpy                     */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t rng[4];
    int64_t done;
    int64_t step;
    int64_t ep;
    double v[MAX_SLOTS];
} State;

/* ------------------------------------------------------------------ */
/* Neural policy                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    int obs, nact;
    int o_w1, o_b1, o_w2, o_b2, o_wp, o_bp, o_wv, o_bv, total;
    double *w, *g, *m, *v;
    long opt_step;
    double h1[HID], h2[HID];
    double x[MAX_OBS];
    double logits[MAX_ACT];
    double probs[MAX_ACT];
    double value;
    double dpre1[HID], dpre2[HID];
} Net;

static void net_layout(Net *n) {
    int o = 0;
    n->o_w1 = o; o += n->obs * HID;
    n->o_b1 = o; o += HID;
    n->o_w2 = o; o += HID * HID;
    n->o_b2 = o; o += HID;
    n->o_wp = o; o += n->nact * HID;
    n->o_bp = o; o += n->nact;
    n->o_wv = o; o += HID;
    n->o_bv = o; o += 1;
    n->total = o;
}

static Net *net_new(int obs, int nact, uint64_t seed) {
    Net *n = (Net *)xcalloc(1, sizeof(Net));
    n->obs = obs;
    n->nact = nact;
    net_layout(n);
    n->w = (double *)xcalloc((size_t)n->total, sizeof(double));
    n->g = (double *)xcalloc((size_t)n->total, sizeof(double));
    n->m = (double *)xcalloc((size_t)n->total, sizeof(double));
    n->v = (double *)xcalloc((size_t)n->total, sizeof(double));

    uint64_t rng[4];
    rng_seed(rng, seed ^ 0xA5A5A5A5A5A5A5A5ULL);
    double s1 = sqrt(2.0 / (double)(obs > 0 ? obs : 1));
    double s2 = sqrt(2.0 / (double)HID);
    for (int i = 0; i < HID; i++)
        for (int j = 0; j < obs; j++)
            n->w[n->o_w1 + i * obs + j] = (rng_double(rng) * 2 - 1) * s1;
    for (int i = 0; i < HID; i++)
        for (int j = 0; j < HID; j++)
            n->w[n->o_w2 + i * HID + j] = (rng_double(rng) * 2 - 1) * s2;
    for (int a = 0; a < nact; a++)
        for (int i = 0; i < HID; i++)
            n->w[n->o_wp + a * HID + i] = (rng_double(rng) * 2 - 1) * 0.01;
    for (int i = 0; i < HID; i++)
        n->w[n->o_wv + i] = (rng_double(rng) * 2 - 1) * 0.01;
    return n;
}

static void net_free(Net *n) {
    if (!n) return;
    free(n->w); free(n->g); free(n->m); free(n->v); free(n);
}

static void net_forward(Net *n, const double *obs) {
    memcpy(n->x, obs, sizeof(double) * (size_t)n->obs);
    const double *w = n->w;
    for (int i = 0; i < HID; i++) {
        double acc = w[n->o_b1 + i];
        const double *row = w + n->o_w1 + i * n->obs;
        for (int j = 0; j < n->obs; j++) acc += row[j] * n->x[j];
        n->h1[i] = tanh(acc);
    }
    for (int i = 0; i < HID; i++) {
        double acc = w[n->o_b2 + i];
        const double *row = w + n->o_w2 + i * HID;
        for (int j = 0; j < HID; j++) acc += row[j] * n->h1[j];
        n->h2[i] = tanh(acc);
    }
    double maxl = -1e30;
    for (int a = 0; a < n->nact; a++) {
        double acc = w[n->o_bp + a];
        const double *row = w + n->o_wp + a * HID;
        for (int i = 0; i < HID; i++) acc += row[i] * n->h2[i];
        n->logits[a] = acc;
        if (acc > maxl) maxl = acc;
    }
    double sum = 0;
    for (int a = 0; a < n->nact; a++) {
        n->probs[a] = exp(n->logits[a] - maxl);
        sum += n->probs[a];
    }
    for (int a = 0; a < n->nact; a++) n->probs[a] /= sum;

    double val = w[n->o_bv];
    for (int i = 0; i < HID; i++) val += w[n->o_wv + i] * n->h2[i];
    n->value = val;
}

/* Every action of an n-action decision, as a legal-move mask. */
static uint32_t mask_all(int n) {
    return n >= 32 ? 0xFFFFFFFFu : (((uint32_t)1 << n) - 1u);
}

static int mask_count(uint32_t mask, int n) {
    int k = 0;
    for (int a = 0; a < n; a++) k += (int)((mask >> a) & 1u);
    return k;
}

/* Restrict the policy to the legal actions, after net_forward. It is redone
 * from the logits instead of dividing the probabilities, because a policy that
 * has put almost everything on an illegal action leaves the legal ones too
 * small to divide. An illegal action gets probability exactly 0, which also
 * zeroes its gradient and its share of the entropy. Does nothing when every
 * action is legal, so an unmasked decision is computed exactly as before. */
static void net_mask(Net *n, uint32_t mask) {
    uint32_t all = mask_all(n->nact);
    mask &= all;
    if (mask == all || mask == 0) return;
    double maxl = -1e30;
    for (int a = 0; a < n->nact; a++)
        if (((mask >> a) & 1u) && n->logits[a] > maxl) maxl = n->logits[a];
    double sum = 0;
    for (int a = 0; a < n->nact; a++) {
        n->probs[a] = ((mask >> a) & 1u) ? exp(n->logits[a] - maxl) : 0.0;
        sum += n->probs[a];
    }
    for (int a = 0; a < n->nact; a++) n->probs[a] /= sum;
}

static double net_entropy(const Net *n) {
    double h = 0;
    for (int a = 0; a < n->nact; a++) {
        double p = n->probs[a];
        if (p > 1e-12) h -= p * log(p);
    }
    return h;
}

static void net_backward(Net *n, const double *dlogits, double dvalue) {
    double *g = n->g;
    for (int a = 0; a < n->nact; a++) {
        g[n->o_bp + a] += dlogits[a];
        for (int i = 0; i < HID; i++)
            g[n->o_wp + a * HID + i] += dlogits[a] * n->h2[i];
    }
    g[n->o_bv] += dvalue;
    for (int i = 0; i < HID; i++) g[n->o_wv + i] += dvalue * n->h2[i];

    for (int i = 0; i < HID; i++) {
        double d = dvalue * n->w[n->o_wv + i];
        for (int a = 0; a < n->nact; a++)
            d += dlogits[a] * n->w[n->o_wp + a * HID + i];
        n->dpre2[i] = d * (1.0 - n->h2[i] * n->h2[i]);
    }
    for (int i = 0; i < HID; i++) {
        g[n->o_b2 + i] += n->dpre2[i];
        for (int j = 0; j < HID; j++)
            g[n->o_w2 + i * HID + j] += n->dpre2[i] * n->h1[j];
    }
    for (int j = 0; j < HID; j++) {
        double d = 0;
        for (int i = 0; i < HID; i++) d += n->dpre2[i] * n->w[n->o_w2 + i * HID + j];
        n->dpre1[j] = d * (1.0 - n->h1[j] * n->h1[j]);
    }
    for (int i = 0; i < HID; i++) {
        g[n->o_b1 + i] += n->dpre1[i];
        for (int j = 0; j < n->obs; j++)
            g[n->o_w1 + i * n->obs + j] += n->dpre1[i] * n->x[j];
    }
}

static void net_clip_grads(Net *n, double max_norm) {
    double sq = 0;
    for (int i = 0; i < n->total; i++) sq += n->g[i] * n->g[i];
    double norm = sqrt(sq);
    if (norm > max_norm && norm > 0) {
        double s = max_norm / norm;
        for (int i = 0; i < n->total; i++) n->g[i] *= s;
    }
}

static void net_adam(Net *n, double lr, double weight_decay) {
    n->opt_step++;
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    double c1 = 1.0 - pow(b1, (double)n->opt_step);
    double c2 = 1.0 - pow(b2, (double)n->opt_step);
    for (int i = 0; i < n->total; i++) {
        double g = n->g[i];
        n->m[i] = b1 * n->m[i] + (1 - b1) * g;
        n->v[i] = b2 * n->v[i] + (1 - b2) * g * g;
        double mh = n->m[i] / c1;
        double vh = n->v[i] / c2;
        n->w[i] -= lr * (mh / (sqrt(vh) + eps) + weight_decay * n->w[i]);
        n->g[i] = 0.0;
    }
}

/* ------------------------------------------------------------------ */
/* Table policy: one row per (agent, observation, legal moves)         */
/* ------------------------------------------------------------------ */

/* Self-play by regret matching keeps, for every situation an agent has been
 * in, how much it regrets not having played each action, and the sum of the
 * strategies it has played there. A situation is what the agent is shown: the
 * observation and which moves are legal. The same rows, read back as
 * probabilities, are the policy the training produces. */
typedef struct {
    long n, cap;
    uint64_t *hash;
    int *agent, *nact, *klen;
    uint32_t *mask;
    long *koff;         /* where the row's observation starts in `keys` */
    long *aoff;         /* where its actions start in `regret` and `ssum` */
    double *keys;
    long nkeys, capkeys;
    double *regret, *ssum;
    long nvals, capvals;
    long *slot;         /* open addressing, -1 = empty */
    long nslot;
} Table;

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static Table *tab_new(void) {
    Table *T = (Table *)xcalloc(1, sizeof(Table));
    T->nslot = 1024;
    T->slot = (long *)xmalloc(sizeof(long) * (size_t)T->nslot);
    for (long i = 0; i < T->nslot; i++) T->slot[i] = -1;
    return T;
}

static void tab_free(Table *T) {
    if (!T) return;
    free(T->hash); free(T->agent); free(T->nact); free(T->klen); free(T->mask);
    free(T->koff); free(T->aoff); free(T->keys); free(T->regret); free(T->ssum);
    free(T->slot); free(T);
}

static uint64_t mix64(uint64_t h, uint64_t x) {
    h ^= x + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return h;
}

static uint64_t bits_of(double v) {
    if (v == 0.0) v = 0.0;      /* -0.0 and 0.0 are one observation */
    uint64_t b;
    memcpy(&b, &v, sizeof(b));
    return b;
}

static uint64_t tab_hash(int agent, const double *obs, int n, uint32_t mask) {
    uint64_t h = mix64(0x1234567ULL + (uint64_t)agent, mask);
    for (int i = 0; i < n; i++) h = mix64(h, bits_of(obs[i]));
    return h;
}

/* The row for this situation, or -1. With `create` a missing row is added
 * with no regrets, which regret matching plays uniformly. */
static long tab_find(Table *T, int agent, const double *obs, int n, int nact,
                     uint32_t mask, int create) {
    uint64_t h = tab_hash(agent, obs, n, mask);
    long at = (long)(h & (uint64_t)(T->nslot - 1));
    for (;;) {
        long id = T->slot[at];
        if (id < 0) break;
        if (T->hash[id] == h && T->agent[id] == agent && T->mask[id] == mask &&
            T->klen[id] == n) {
            const double *k = T->keys + T->koff[id];
            int same = 1;
            for (int i = 0; i < n && same; i++)
                if (bits_of(k[i]) != bits_of(obs[i])) same = 0;
            if (same) return id;
        }
        at = (at + 1) & (T->nslot - 1);
    }
    if (!create) return -1;
    if (T->n >= T->cap) {
        T->cap = T->cap ? T->cap * 2 : 256;
        T->hash = (uint64_t *)xrealloc(T->hash, sizeof(uint64_t) * (size_t)T->cap);
        T->agent = (int *)xrealloc(T->agent, sizeof(int) * (size_t)T->cap);
        T->nact = (int *)xrealloc(T->nact, sizeof(int) * (size_t)T->cap);
        T->klen = (int *)xrealloc(T->klen, sizeof(int) * (size_t)T->cap);
        T->mask = (uint32_t *)xrealloc(T->mask, sizeof(uint32_t) * (size_t)T->cap);
        T->koff = (long *)xrealloc(T->koff, sizeof(long) * (size_t)T->cap);
        T->aoff = (long *)xrealloc(T->aoff, sizeof(long) * (size_t)T->cap);
    }
    if (T->nkeys + n > T->capkeys) {
        T->capkeys = T->capkeys ? T->capkeys * 2 : 4096;
        while (T->nkeys + n > T->capkeys) T->capkeys *= 2;
        T->keys = (double *)xrealloc(T->keys, sizeof(double) * (size_t)T->capkeys);
    }
    if (T->nvals + nact > T->capvals) {
        T->capvals = T->capvals ? T->capvals * 2 : 4096;
        T->regret = (double *)xrealloc(T->regret, sizeof(double) * (size_t)T->capvals);
        T->ssum = (double *)xrealloc(T->ssum, sizeof(double) * (size_t)T->capvals);
    }
    long id = T->n++;
    T->hash[id] = h;
    T->agent[id] = agent;
    T->nact[id] = nact;
    T->klen[id] = n;
    T->mask[id] = mask;
    T->koff[id] = T->nkeys;
    T->aoff[id] = T->nvals;
    memcpy(T->keys + T->nkeys, obs, sizeof(double) * (size_t)n);
    T->nkeys += n;
    for (int a = 0; a < nact; a++) {
        T->regret[T->nvals + a] = 0.0;
        T->ssum[T->nvals + a] = 0.0;
    }
    T->nvals += nact;
    T->slot[at] = id;
    if (T->n * 2 > T->nslot) {
        long ns = T->nslot * 2;
        long *slot = (long *)xmalloc(sizeof(long) * (size_t)ns);
        for (long i = 0; i < ns; i++) slot[i] = -1;
        for (long k = 0; k < T->n; k++) {
            long j = (long)(T->hash[k] & (uint64_t)(ns - 1));
            while (slot[j] >= 0) j = (j + 1) & (ns - 1);
            slot[j] = k;
        }
        free(T->slot);
        T->slot = slot;
        T->nslot = ns;
    }
    return id;
}

static void probs_uniform(double *probs, int nact, uint32_t legal) {
    int k = 0;
    for (int a = 0; a < nact; a++) k += (int)((legal >> a) & 1u);
    for (int a = 0; a < nact; a++)
        probs[a] = ((legal >> a) & 1u) ? 1.0 / (double)(k ? k : 1) : 0.0;
}

/* Regret matching: play each legal action in proportion to its positive
 * regret, or uniformly when there is none. */
static void tab_current(const Table *T, long id, double *probs) {
    int n = T->nact[id];
    uint32_t legal = T->mask[id];
    const double *r = T->regret + T->aoff[id];
    double sum = 0.0;
    for (int a = 0; a < n; a++)
        if (((legal >> a) & 1u) && r[a] > 0.0) sum += r[a];
    if (!(sum > 0.0)) { probs_uniform(probs, n, legal); return; }
    for (int a = 0; a < n; a++)
        probs[a] = (((legal >> a) & 1u) && r[a] > 0.0) ? r[a] / sum : 0.0;
}

/* The average of the strategies played here, which is what regret matching
 * converges through. A row that was never averaged falls back to its current
 * strategy. */
static void tab_average(const Table *T, long id, double *probs) {
    int n = T->nact[id];
    uint32_t legal = T->mask[id];
    const double *s = T->ssum + T->aoff[id];
    double sum = 0.0;
    for (int a = 0; a < n; a++)
        if (((legal >> a) & 1u) && s[a] > 0.0) sum += s[a];
    if (!(sum > 0.0)) { tab_current(T, id, probs); return; }
    for (int a = 0; a < n; a++)
        probs[a] = (((legal >> a) & 1u) && s[a] > 0.0) ? s[a] / sum : 0.0;
}

enum { TAB_OFF = 0, TAB_CURRENT, TAB_AVERAGE };

/* ------------------------------------------------------------------ */
/* VM                                                                  */
/* ------------------------------------------------------------------ */

struct VM;

/* A driver takes decisions and chance away from the policy and the random
 * stream: a walk over every branch, a plan being replayed, a person at the
 * keyboard. Each hook returns the outcome to use, DRIVE_PASS to leave it to
 * the policy or the stream, or DRIVE_STOP to end the block right there. */
#define DRIVE_PASS (-1)
#define DRIVE_STOP (-2)

typedef struct Driver {
    int (*decide)(struct Driver *dr, struct VM *vm, int agent, const double *obs,
                  int obs_len, int n_act, uint32_t legal, const double *probs);
    int (*chance)(struct Driver *dr, struct VM *vm, int lo, int count);
    /* Told the action every decision ended with, whoever chose it. */
    void (*decided)(struct Driver *dr, struct VM *vm, int agent, int n_act, uint32_t legal,
                    const double *probs, int action);
    int no_rand;        /* a rand() cannot be enumerated: stop and say so */
} Driver;

typedef struct {
    int agent;
    int action;
    double logp;
    double value;
    double prob[MAX_ACT];
    double obs[MAX_OBS];
    int obs_dim;
    int n_act;
    uint32_t mask;      /* bit a set = action a was legal; `prob` is already masked */
    /* A decision made by a search: how many playouts tried each action and
     * the mean return they saw. `prob` is then 1 for the action played. */
    int searched;
    int tries[MAX_ACT];
    double mean[MAX_ACT];
} Decision;

/* What `hunch test` reports beyond pass or fail: how often each assert ran,
 * and each place where the language's forgiving rules (clamp an index, divide
 * by zero to get 0) were actually used. Keyed by kind and source position. */
#define MAX_NOTES 256

enum { NOTE_ASSERT = 0, NOTE_INDEX_READ, NOTE_INDEX_WRITE, NOTE_DIV_ZERO, NOTE_MOD_ZERO };

typedef struct { int kind, line, col; long count; } Note;

typedef struct {
    Note d[MAX_NOTES];
    int n;
    long lost;          /* events dropped because the table was full */
} Notes;

/* The episode `trace` is playing, so a failed assert can say where it was. */
static int g_trace_on = 0;
static uint64_t g_trace_seed = 0;

struct Search;
static int search_choose(struct Search *S, struct VM *vm, int ip, int agent, int n_act,
                         uint32_t legal, const double *obs, int obs_len);
static void search_found(const struct Search *S, int *tries, double *mean);

typedef struct VM {
    const Program *p;
    State *s;
    State branch;
    Net *nets[MAX_AGENTS];
    /* When set, neither the table nor the networks decide: the world is
     * forked at each decision and searched (see "Search" below). */
    struct Search *search;
    int in_init;        /* the block being run is init, not a step */
    /* When tab_mode is not TAB_OFF the table decides, not the networks. */
    Table *tab;
    int tab_mode;
    int tab_create;     /* add a row for a situation the table has not met */
    Driver *driver;
    int stopped;        /* the driver ended the block: 1 at an event, 2 at a rand() */
    long tab_row;       /* the table row of the decision being made, or -1 */
    long tab_seen, tab_missed;  /* decisions the table was asked, and had no row for */
    int rand_line, rand_col;    /* where a rand() stopped a driver that cannot take one */
    int deterministic;
    int record;
    int n_dec;
    Decision dec[MAX_DEC_PER_STEP];
    double agent_reward[MAX_AGENTS];
    int force_agent;
    int force_action;
    double scratch[MAX_SCRATCH];
    double stack[STACK_SIZE];
    int sp;
    int call_ip[CALL_DEPTH];
    int csp;
    int at_ip;          /* the instruction after the decision being asked */
    int halt;
    long insns;
    long total_steps;
    long max_steps;
    double gamma;
    /* scripted actions for one agent (used by interactive play) */
    int script_agent;
    const int *script;
    /* ask: when the script runs out at a decision of its agent, stop there and
     * say what that agent is shown, instead of letting the policy play on. */
    int ask;
    int ask_agent, ask_n, ask_obs_len;
    uint32_t ask_legal;
    uint32_t random_agents;     /* bit a set = agent a plays uniformly, whatever its policy */
    int script_n;
    int script_pos;
    int freeze_agent;   /* -1 = none: this agent always takes freeze_action */
    int freeze_action;
    int record_agent;   /* -1 = all: only record decisions for this agent */
    int fork_horizon;   /* how far a counterfactual branch rolls before bootstrapping */
    /* Legal-move mask left by OP_LEGAL for the OP_HUNCH right after it. */
    uint32_t legal_mask;
    int legal_pending;
    int legal_line, legal_col;
    /* Set when a forced action turned out to be illegal where it landed. */
    int force_missed;
    /* strict: also note clamped indices and zero divisors (assert runs are
     * always counted). */
    int strict;
    Notes notes;
} VM;

static void vm_init(VM *vm, const Program *p, State *s) {
    memset(vm, 0, sizeof(*vm));
    vm->p = p;
    vm->s = s;
    vm->force_agent = -1;
    vm->script_agent = -1;
    vm->freeze_agent = -1;
    vm->record_agent = -1;
    vm->fork_horizon = 24;
    vm->gamma = 0.99;
}

static double *varp(VM *vm, int idx) {
    if (idx < 0) idx = 0;
    if (idx >= vm->p->nslots) idx = vm->p->nslots - 1;
    return &vm->s->v[idx];
}

static void push(VM *vm, double v) {
    if (vm->sp >= STACK_SIZE - 1) { vm->halt = 1; return; }
    vm->stack[vm->sp++] = v;
}

static double pop(VM *vm) {
    if (vm->sp <= 0) return 0.0;
    return vm->stack[--vm->sp];
}

/* `*bad` is set when the index had to be clamped (or was NaN). The result is
 * the same either way; only `hunch test` wants to know. */
static int pop_index(VM *vm, int len, int *bad) {
    double d = pop(vm);
    if (!(d == d)) { *bad = 1; return 0; }
    int i = (int)floor(d);
    if (i < 0) { i = 0; *bad = 1; }
    if (i >= len) { i = len - 1; *bad = 1; }
    return i;
}

/* Count one event at a source position. The table is scanned in order: a
 * world has a handful of asserts, and the other kinds are only recorded under
 * `strict`. Once it is full, new positions are counted in `lost` so the
 * report can say it is incomplete instead of looking clean. */
static void vm_note(VM *vm, int kind, int line, int col) {
    Notes *t = &vm->notes;
    for (int i = 0; i < t->n; i++) {
        Note *e = &t->d[i];
        if (e->kind == kind && e->line == line && e->col == col) {
            e->count++;
            return;
        }
    }
    if (t->n >= MAX_NOTES) { t->lost++; return; }
    Note *e = &t->d[t->n++];
    e->kind = kind;
    e->line = line;
    e->col = col;
    e->count = 1;
}

/* One JSON line: every assert in the program with how often it ran, then
 * every note. The asserts are found by scanning the bytecode, so one that
 * never ran is listed with 0 runs instead of being left out. */
static void vm_report(const VM *vm, FILE *f) {
    static const char *names[] = {
        "assert", "index_read", "index_write", "div_zero", "mod_zero"
    };
    const Program *p = vm->p;
    const Notes *t = &vm->notes;
    int first = 1;
    fputs("{\"asserts\":[", f);
    for (int ip = 0; ip < p->ncode; ip++) {
        const int32_t *ins = p->code + (size_t)ip * 4;
        if (ins[0] != OP_ASSERT) continue;
        long runs = 0;
        for (int i = 0; i < t->n; i++) {
            const Note *e = &t->d[i];
            if (e->kind == NOTE_ASSERT && e->line == ins[1] && e->col == ins[2])
                runs = e->count;
        }
        fprintf(f, "%s{\"line\":%d,\"col\":%d,\"runs\":%ld}", first ? "" : ",",
                ins[1], ins[2], runs);
        first = 0;
    }
    fputs("],\"notes\":[", f);
    first = 1;
    for (int i = 0; i < t->n; i++) {
        const Note *e = &t->d[i];
        if (e->kind == NOTE_ASSERT) continue;
        fprintf(f, "%s{\"kind\":\"%s\",\"line\":%d,\"col\":%d,\"count\":%ld}",
                first ? "" : ",", names[e->kind], e->line, e->col, e->count);
        first = 0;
    }
    fputs("]", f);
    if (t->lost) fprintf(f, ",\"lost\":%ld", t->lost);
    fputs("}\n", f);
}

/* A decision that cannot be made as asked. Like a failed assert, this is the
 * world or the caller breaking a rule, so the run stops and says where. */
static void hunch_fail(const VM *vm, int agent, const char *what, int action) {
    const char *name = vm->p->agents[agent].name;
    if (action >= 0)
        fatal(3, "hunch: %s action %d is illegal for '%s' at line %d, column %d",
              what, action, name, vm->legal_line, vm->legal_col);
    fatal(3, "hunch: no legal action for '%s' at line %d, column %d",
          name, vm->legal_line, vm->legal_col);
}

/* Run from instruction `ip` with the stack, the calls and the counters as
 * they stand: a block from its start (vm_run), or the rest of one from a
 * decision, which is how a search continues a copy of a block in its own VM. */
static void vm_exec(VM *vm, int ip) {
    const Program *p = vm->p;
    const int32_t *code = p->code;

    for (;;) {
        if (++vm->insns > INSN_BUDGET) { vm->halt = 1; break; }
        const int32_t *ins = code + (size_t)ip * 4;
        int op = ins[0];
        int a = ins[1], b = ins[2], c = ins[3];
        ip++;
        switch (op) {
        case OP_HALT: goto done;
        case OP_PUSH_CONST: push(vm, p->consts[a]); break;
        case OP_PUSH_VAR: push(vm, *varp(vm, a)); break;
        case OP_STORE_VAR: *varp(vm, a) = pop(vm); break;
        /* The third operand of the two index ops is the source position,
         * line * 4096 + column. It only matters to a strict run. */
        case OP_PUSH_IDX: {
            int bad = 0;
            int i = pop_index(vm, b, &bad);
            if (bad && vm->strict) vm_note(vm, NOTE_INDEX_READ, c >> 12, c & 4095);
            push(vm, *varp(vm, a + i));
            break;
        }
        case OP_STORE_IDX: {
            int bad = 0;
            int i = pop_index(vm, b, &bad);
            double v = pop(vm);
            if (bad && vm->strict) vm_note(vm, NOTE_INDEX_WRITE, c >> 12, c & 4095);
            *varp(vm, a + i) = v;
            break;
        }
        case OP_ASSERT:
            vm_note(vm, NOTE_ASSERT, a, b);
            if (pop(vm) == 0.0) {
                if (g_searching)
                    fatal(3, "hunch: assertion failed at line %d, column %d\n"
                          "  in step %d of a continuation the search was trying, not in the "
                          "episode itself", a, b, (int)vm->s->step);
                if (g_trace_on && vm->s->step == 0)
                    fatal(3, "hunch: assertion failed at line %d, column %d\n"
                          "  in init, in the episode with seed %llu", a, b,
                          (unsigned long long)g_trace_seed);
                else if (g_trace_on)
                    fatal(3, "hunch: assertion failed at line %d, column %d\n"
                          "  in step %d of the episode with seed %llu", a, b,
                          (int)vm->s->step, (unsigned long long)g_trace_seed);
                fatal(3, "hunch: assertion failed at line %d, column %d", a, b);
            }
            break;
        case OP_POP: pop(vm); break;
        case OP_DUP: { double v = pop(vm); push(vm, v); push(vm, v); break; }
        case OP_ADD: { double y = pop(vm), x = pop(vm); push(vm, x + y); break; }
        case OP_SUB: { double y = pop(vm), x = pop(vm); push(vm, x - y); break; }
        case OP_MUL: { double y = pop(vm), x = pop(vm); push(vm, x * y); break; }
        /* DIV and MOD carry their source position in a and b for the same reason. */
        case OP_DIV: {
            double y = pop(vm), x = pop(vm);
            if (y == 0.0 && vm->strict) vm_note(vm, NOTE_DIV_ZERO, a, b);
            push(vm, y == 0.0 ? 0.0 : x / y);
            break;
        }
        case OP_MOD: {
            double y = pop(vm), x = pop(vm);
            if (y == 0.0 && vm->strict) vm_note(vm, NOTE_MOD_ZERO, a, b);
            push(vm, y == 0.0 ? 0.0 : fmod(x, y));
            break;
        }
        case OP_NEG: push(vm, -pop(vm)); break;
        case OP_EQ: { double y = pop(vm), x = pop(vm); push(vm, x == y); break; }
        case OP_NE: { double y = pop(vm), x = pop(vm); push(vm, x != y); break; }
        case OP_LT: { double y = pop(vm), x = pop(vm); push(vm, x < y); break; }
        case OP_LE: { double y = pop(vm), x = pop(vm); push(vm, x <= y); break; }
        case OP_GT: { double y = pop(vm), x = pop(vm); push(vm, x > y); break; }
        case OP_GE: { double y = pop(vm), x = pop(vm); push(vm, x >= y); break; }
        case OP_AND: { double y = pop(vm), x = pop(vm); push(vm, (x != 0 && y != 0)); break; }
        case OP_OR: { double y = pop(vm), x = pop(vm); push(vm, (x != 0 || y != 0)); break; }
        case OP_NOT: push(vm, pop(vm) == 0 ? 1.0 : 0.0); break;
        case OP_JUMP: ip = a; break;
        case OP_JUMP_FALSE: if (pop(vm) == 0.0) ip = a; break;
        case OP_CALL: {
            if (vm->csp >= CALL_DEPTH) { vm->halt = 1; break; }
            const FuncInfo *f = &p->funcs[a];
            /* Arguments were pushed left to right, so pop them back to front. */
            for (int i = f->nparams - 1; i >= 0; i--)
                *varp(vm, f->frame_base + i) = pop(vm);
            vm->call_ip[vm->csp] = ip;
            vm->csp++;
            ip = f->entry;
            break;
        }
        case OP_RET: {
            double v = pop(vm);
            if (vm->csp <= 0) { vm->halt = 1; break; }
            vm->csp--;
            ip = vm->call_ip[vm->csp];
            push(vm, v);
            break;
        }
        case OP_SCRATCH: vm->scratch[a] = pop(vm); break;
        case OP_COPY_VARS:
            for (int i = 0; i < b; i++) vm->scratch[c + i] = *varp(vm, a + i);
            break;
        case OP_ZERO:
            for (int i = 0; i < b; i++) *varp(vm, a + i) = 0.0;
            break;
        case OP_LEGAL: {
            /* The values were pushed for action 0 first, so they come off the
             * stack back to front. The mask waits for the OP_HUNCH that the
             * compiler always puts next. */
            uint32_t m = 0;
            for (int i = a - 1; i >= 0; i--) {
                double v = pop(vm);
                if (v != 0.0 && i < MAX_ACT) m |= (uint32_t)1 << i;
            }
            vm->legal_mask = m;
            vm->legal_pending = 1;
            vm->legal_line = b;
            vm->legal_col = c;
            break;
        }
        case OP_HUNCH: {
            int agent = a, obs_len = b, n_act = c;
            int masked = vm->legal_pending;
            vm->legal_pending = 0;
            if (agent < 0 || agent >= p->nagents || n_act < 1 || n_act > MAX_ACT) {
                push(vm, 0);
                break;
            }
            /* Without a mask every action is legal, and everything below then
             * computes exactly what it did before masks existed. */
            uint32_t all = mask_all(n_act);
            uint32_t legal = masked ? (vm->legal_mask & all) : all;
            if (legal == 0) hunch_fail(vm, agent, NULL, -1);
            int at_random = (int)((vm->random_agents >> agent) & 1u);
            Net *net = at_random ? NULL : vm->nets[agent];
            /* A search has no probabilities until it has run, and it runs
             * only if the choice is left to the policy: a driver's `decide`
             * is shown uniform ones. */
            struct Search *search = at_random ? NULL : vm->search;
            int searched = 0;
            Decision *d = NULL;
            if (vm->record && vm->n_dec < MAX_DEC_PER_STEP &&
                (vm->record_agent < 0 || vm->record_agent == agent)) {
                d = &vm->dec[vm->n_dec];
                memset(d, 0, sizeof(*d));
                d->agent = agent;
                d->obs_dim = obs_len;
                d->n_act = n_act;
                d->mask = legal;
                memcpy(d->obs, vm->scratch, sizeof(double) * (size_t)obs_len);
            }

            double probs[MAX_ACT];
            double value = 0.0;
            int by_table = vm->tab_mode != TAB_OFF && vm->tab && !at_random;
            vm->tab_row = -1;
            if (by_table) {
                long row = tab_find(vm->tab, agent, vm->scratch, obs_len, n_act, legal,
                                    vm->tab_create);
                vm->tab_row = row;
                vm->tab_seen++;
                if (row < 0) vm->tab_missed++;
                if (row < 0) probs_uniform(probs, n_act, legal);
                else if (vm->tab_mode == TAB_CURRENT) tab_current(vm->tab, row, probs);
                else tab_average(vm->tab, row, probs);
            } else if (net) {
                net_forward(net, vm->scratch);
                net_mask(net, legal);
                memcpy(probs, net->probs, sizeof(double) * (size_t)n_act);
                value = net->value;
            } else {
                probs_uniform(probs, n_act, legal);
            }
            int driven = DRIVE_PASS;
            if (vm->driver && vm->driver->decide) {
                vm->at_ip = ip;
                driven = vm->driver->decide(vm->driver, vm, agent, vm->scratch, obs_len,
                                            n_act, legal, probs);
                if (driven == DRIVE_STOP) {
                    vm->stopped = 1;
                    vm->halt = 1;
                    break;
                }
            }

            /* One draw per decision, always, even when the action is forced.
             * A world where two agents decide on the same tick relies on this:
             * if a forced branch skipped its draw, every later random number
             * would shift and the branches would stop being comparable. */
            double u = rng_double(vm->s->rng);

            /* The three policy branches walk the legal actions only, so they
             * cannot land on an illegal one even when rounding leaves the
             * draw past the last probability. */
            int action = -1;
            const char *asked = NULL;   /* who chose, when it was not the policy */
            if (driven >= 0) {
                action = driven;
                asked = "chosen";
            } else if (vm->freeze_agent == agent) {
                action = vm->freeze_action;
                asked = "frozen";
            } else if (vm->script_agent == agent && vm->script_pos < vm->script_n) {
                action = vm->script[vm->script_pos++];
                asked = "scripted";
            } else if (vm->script_agent == agent && vm->ask) {
                vm->ask_agent = agent;
                vm->ask_n = n_act;
                vm->ask_obs_len = obs_len;
                vm->ask_legal = legal;
                vm->stopped = 3;
                vm->halt = 1;
                break;
            } else if (vm->force_agent == agent) {
                action = vm->force_action;
                vm->force_agent = -1;
                if (action < 0) action = 0;
                if (action >= n_act) action = n_act - 1;
                /* A fork branch replays the step from its start, and the world
                 * can reach this decision in a different state than the one the
                 * action was picked for: another agent decided first with a
                 * policy that has been updated since, or this agent decides
                 * twice in a step. That is not the world's mistake, so it is
                 * not an error. The branch measures nothing; say so and let
                 * the policy choose, so the step still finishes sanely. */
                if (!((legal >> action) & 1u)) {
                    vm->force_missed = 1;
                    action = -1;
                }
            }
            if (asked) {
                /* Chosen by the caller, who is told if it cannot be played. */
                if (action < 0) action = 0;
                if (action >= n_act) action = n_act - 1;
                if (!((legal >> action) & 1u)) hunch_fail(vm, agent, asked, action);
            } else if (action >= 0) {
                /* forced, and legal here */
            } else if (search) {
                /* The draw above is the only one taken from the world's
                 * stream: the search has its own, started from the instant. */
                vm->at_ip = ip;
                action = search_choose(search, vm, ip, agent, n_act, legal, vm->scratch, obs_len);
                for (int i = 0; i < n_act; i++) probs[i] = i == action ? 1.0 : 0.0;
                searched = 1;
            } else if (!net && !by_table) {
                /* Uniform: the k-th legal action. */
                int k = (int)(u * mask_count(legal, n_act));
                for (int i = 0; i < n_act; i++) {
                    if (!((legal >> i) & 1u)) continue;
                    action = i;
                    if (k-- == 0) break;
                }
            } else if (vm->deterministic) {
                for (int i = 0; i < n_act; i++) {
                    if (!((legal >> i) & 1u)) continue;
                    if (action < 0 || probs[i] > probs[action]) action = i;
                }
            } else {
                double acc = 0;
                for (int i = 0; i < n_act; i++) {
                    if (!((legal >> i) & 1u)) continue;
                    action = i;
                    acc += probs[i];
                    if (u < acc) break;
                }
            }

            if (vm->driver && vm->driver->decided)
                vm->driver->decided(vm->driver, vm, agent, n_act, legal, probs, action);
            if (d) {
                d->action = action;
                d->value = value;
                d->logp = log(probs[action] > 1e-12 ? probs[action] : 1e-12);
                memcpy(d->prob, probs, sizeof(double) * (size_t)n_act);
                if (searched) {
                    d->searched = 1;
                    search_found(search, d->tries, d->mean);
                }
                vm->n_dec++;
            }
            push(vm, (double)action);
            break;
        }
        case OP_REWARD: {
            double v = pop(vm);
            if (a >= 0 && a < MAX_AGENTS) vm->agent_reward[a] += v;
            break;
        }
        case OP_DONE:
            vm->s->done = 1;
            vm->halt = 1;
            break;
        case OP_RAND:
            if (vm->driver && vm->driver->no_rand) {
                vm->rand_line = a;
                vm->rand_col = b;
                vm->stopped = 2;
                vm->halt = 1;
                break;
            }
            push(vm, rng_double(vm->s->rng));
            break;
        case OP_RANDINT: {
            double hi = pop(vm), lo = pop(vm);
            int l = (int)floor(lo), h = (int)floor(hi);
            if (h < l) { int t = l; l = h; h = t; }
            /* One draw per randint, always, even when a driver names the
             * outcome, so a replay stays on the same stream. */
            double u = rng_double(vm->s->rng);
            double span = (double)h - (double)l + 1.0;      /* h - l + 1 can pass an int */
            int v = (int)((double)l + floor(u * span));
            if (v > h) v = h;
            if (vm->driver && vm->driver->no_rand && span > MAX_LISTED) {
                /* Too many outcomes to follow one by one: like a rand(). */
                vm->rand_line = a;
                vm->rand_col = -1 - b;      /* negative: a randint, not a rand() */
                vm->stopped = 2;
                vm->halt = 1;
                break;
            }
            if (vm->driver && vm->driver->chance) {
                int got = vm->driver->chance(vm->driver, vm, l,
                                             span > 2147483647.0 ? 2147483647 : (int)span);
                if (got == DRIVE_STOP) {
                    vm->stopped = 1;
                    vm->halt = 1;
                    break;
                }
                if (got >= 0) v = (double)got > (double)h - (double)l ? h : l + got;
            }
            push(vm, (double)v);
            break;
        }
        case OP_ABS: push(vm, fabs(pop(vm))); break;
        case OP_MIN: { double y = pop(vm), x = pop(vm); push(vm, x < y ? x : y); break; }
        case OP_MAX: { double y = pop(vm), x = pop(vm); push(vm, x > y ? x : y); break; }
        case OP_CLAMP: {
            double hi = pop(vm), lo = pop(vm), x = pop(vm);
            if (x < lo) x = lo;
            if (x > hi) x = hi;
            push(vm, x);
            break;
        }
        case OP_FLOOR: push(vm, floor(pop(vm))); break;
        case OP_SQRT: { double x = pop(vm); push(vm, x <= 0 ? 0 : sqrt(x)); break; }
        case OP_EXP: {
            double x = pop(vm);
            if (x > 60) x = 60;
            if (x < -60) x = -60;
            push(vm, exp(x));
            break;
        }
        case OP_STEP_COUNT: push(vm, (double)vm->s->step); break;
        default: vm->halt = 1; break;
        }
        if (vm->halt) break;
    }
done:
    vm->sp = 0;
    vm->csp = 0;
}

static void vm_run(VM *vm, int entry) {
    vm->sp = 0;
    vm->csp = 0;
    vm->halt = 0;
    vm->insns = 0;
    vm->stopped = 0;
    /* A block cut off by the instruction budget between OP_LEGAL and its
     * OP_HUNCH must not leave its mask for the next block's first decision. */
    vm->legal_pending = 0;
    vm->in_init = entry == vm->p->init_entry;
    vm_exec(vm, entry);
}

static void state_reset(const Program *p, State *s, uint64_t seed) {
    memcpy(s->v, p->defaults, sizeof(double) * (size_t)p->nslots);
    memset(s->v + p->nslots, 0, sizeof(double) * (size_t)(MAX_SLOTS - p->nslots));
    s->done = 0;
    s->step = 0;
    rng_seed(s->rng, seed);
}

static int vm_step(VM *vm) {
    if (vm->max_steps > 0 && vm->total_steps >= vm->max_steps) return 0;
    vm->s->step++;
    vm->n_dec = 0;
    memset(vm->agent_reward, 0, sizeof(vm->agent_reward));
    vm->sp = 0;
    /* Only `state` lasts from one step to the next. Every local starts the
     * step at zero, so two instants with the same state and step number have
     * the same future, which is what lets the exact graph merge them. */
    if (vm->p->nslots > vm->p->nstate)
        memset(vm->s->v + vm->p->nstate, 0,
               sizeof(double) * (size_t)(vm->p->nslots - vm->p->nstate));
    vm_run(vm, vm->p->step_entry);
    vm->force_agent = -1;
    vm->total_steps++;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Trajectory                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    int agent, ep, action;
    double logp, value, reward, adv, ret, logp_old, rtg;
    double obs[MAX_OBS];
    double prob[MAX_ACT];
    int obs_dim, n_act;
    int valid;
    uint32_t mask;      /* legal actions at this decision; the updates reapply it */
} Step;

typedef struct { Step *d; int n, cap; } Traj;

static void traj_init(Traj *t, int cap) {
    t->d = (Step *)xcalloc((size_t)cap, sizeof(Step));
    t->n = 0;
    t->cap = cap;
}

static void batch_init(Traj *traj, int nagents, int cap) {
    for (int i = 0; i < nagents; i++) traj_init(&traj[i], cap);
}

static void batch_clear(Traj *traj, int nagents) {
    for (int i = 0; i < nagents; i++) traj[i].n = 0;
}

static void batch_free(Traj *traj, int nagents) {
    for (int i = 0; i < nagents; i++) free(traj[i].d);
}

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *algo;
    long budget;
    int horizon;
    double lr;
    double gamma;
    double lambda;
    double clip;
    double entropy;
    double vf_coef;
    double fork_ratio;
    double fork_decay;      /* fraction of the budget over which forking fades out */
    int harvest;            /* keep branch decisions as extra training data */
    int fork_pg;            /* 1 = counterfactual policy gradient, 0 = critic only */
    int fork_inject;        /* 1 = overwrite the taken action's advantage instead */
    int fork_max_points;
    int fork_horizon;
    int ppo_epochs;
    int minibatch;
    int rollout_steps;
    int eval_episodes;
    int verbose;
} Cfg;

static void cfg_defaults(Cfg *c) {
    memset(c, 0, sizeof(*c));
    c->algo = "ppo";
    c->budget = 200000;
    c->horizon = 0;
    c->lr = 3e-3;
    c->gamma = 0.99;
    c->lambda = 0.95;
    c->clip = 0.2;
    c->entropy = 0.01;
    c->vf_coef = 0.5;
    c->fork_ratio = 1.5;
    c->fork_decay = 0.5;
    c->harvest = 0;
    c->fork_pg = 1;
    c->fork_inject = 1;
    c->fork_max_points = 96;
    c->fork_horizon = 24;
    c->ppo_epochs = 4;
    c->minibatch = 256;
    c->rollout_steps = 2048;
    c->eval_episodes = 128;
    c->verbose = 1;
}

/* ------------------------------------------------------------------ */
/* Episode runner                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    State *snap;
    int *agent;
    int *action;
    int *step_idx;
    int *traj_idx;
    int n, cap;
} SnapLog;

static void snaplog_init(SnapLog *sl, int cap) {
    sl->snap = (State *)xmalloc(sizeof(State) * (size_t)cap);
    sl->agent = (int *)xmalloc(sizeof(int) * (size_t)cap);
    sl->action = (int *)xmalloc(sizeof(int) * (size_t)cap);
    sl->step_idx = (int *)xmalloc(sizeof(int) * (size_t)cap);
    sl->traj_idx = (int *)xmalloc(sizeof(int) * (size_t)cap);
    sl->n = 0;
    sl->cap = cap;
}

static void snaplog_free(SnapLog *sl) {
    free(sl->snap); free(sl->agent); free(sl->action);
    free(sl->step_idx); free(sl->traj_idx);
}

/* Run one episode. If record, append decisions to traj and snapshots to sl.
 * ret (optional) receives the undiscounted total reward per agent. */
static void run_episode(VM *vm, Traj *traj, SnapLog *sl, uint64_t seed,
                        int record, int ep, double *ret, int horizon) {
    const Program *p = vm->p;
    State *s = vm->s;
    state_reset(p, s, seed);
    s->ep = ep;
    vm->record = 0;
    vm->force_agent = -1;
    vm->deterministic = 0;
    vm_run(vm, p->init_entry);

    int last_idx[MAX_AGENTS];
    for (int i = 0; i < MAX_AGENTS; i++) last_idx[i] = -1;
    if (ret) for (int i = 0; i < MAX_AGENTS; i++) ret[i] = 0.0;

    for (int t = 0; t < horizon; t++) {
        if (s->done) break;
        State snap;
        if (sl) snap = *s;
        vm->record = record;
        if (!vm_step(vm)) { vm->record = 0; break; }
        vm->record = 0;

        if (ret)
            for (int a = 0; a < p->nagents; a++) ret[a] += vm->agent_reward[a];

        for (int k = 0; k < vm->n_dec; k++) {
            Decision *d = &vm->dec[k];
            if (traj) {
                Traj *tr = &traj[d->agent];
                if (tr->n < tr->cap) {
                    Step *st = &tr->d[tr->n];
                    memset(st, 0, sizeof(*st));
                    st->agent = d->agent;
                    st->ep = ep;
                    st->action = d->action;
                    st->logp = d->logp;
                    st->logp_old = d->logp;
                    st->value = d->value;
                    memcpy(st->obs, d->obs, sizeof(double) * (size_t)d->obs_dim);
                    memcpy(st->prob, d->prob, sizeof(double) * (size_t)d->n_act);
                    st->obs_dim = d->obs_dim;
                    st->n_act = d->n_act;
                    st->mask = d->mask;
                    st->valid = 1;
                    last_idx[d->agent] = tr->n;
                    tr->n++;
                }
            }
            if (sl && sl->n < sl->cap) {
                sl->snap[sl->n] = snap;
                sl->agent[sl->n] = d->agent;
                sl->action[sl->n] = d->action;
                sl->step_idx[sl->n] = t;
                sl->traj_idx[sl->n] = last_idx[d->agent];
                sl->n++;
            }
        }

        if (traj) {
            for (int a = 0; a < p->nagents; a++) {
                if (vm->agent_reward[a] != 0.0 && last_idx[a] >= 0)
                    traj[a].d[last_idx[a]].reward += vm->agent_reward[a];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Advantage estimation                                                */
/* ------------------------------------------------------------------ */

static void compute_gae(Traj *t, double gamma, double lambda) {
    for (int i = t->n - 1; i >= 0; i--) {
        Step *s = &t->d[i];
        int terminal = (i == t->n - 1) || (t->d[i + 1].ep != s->ep);
        double nextv = terminal ? 0.0 : t->d[i + 1].value;
        double delta = s->reward + (terminal ? 0.0 : gamma * nextv) - s->value;
        s->adv = delta + ((terminal || i == t->n - 1) ? 0.0 : gamma * lambda * t->d[i + 1].adv);
        s->ret = s->adv + s->value;
    }
}

/* Discounted return-to-go for every recorded step. This is exactly the value
 * of the action that was actually taken (same universe, same random draws),
 * so the fork pass never has to re-roll it. */
static void compute_rtg(Traj *t, double gamma) {
    double g = 0.0;
    for (int i = t->n - 1; i >= 0; i--) {
        Step *s = &t->d[i];
        int terminal = (i == t->n - 1) || (t->d[i + 1].ep != s->ep);
        if (terminal) g = 0.0;
        g = s->reward + gamma * g;
        s->rtg = g;
    }
}

static void compute_mc(Traj *t, double gamma) {
    for (int i = t->n - 1; i >= 0; i--) {
        Step *s = &t->d[i];
        int terminal = (i == t->n - 1) || (t->d[i + 1].ep != s->ep);
        s->ret = s->reward + (terminal ? 0.0 : gamma * t->d[i + 1].ret);
        s->adv = s->ret;
    }
}

/* ------------------------------------------------------------------ */
/* Policy gradient update                                              */
/* ------------------------------------------------------------------ */

static void policy_update(Net *net, Traj *t, const Cfg *cfg, int n_epochs, int use_clip) {
    int n = t->n;
    if (n < 2) return;

    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mean += t->d[i].adv;
    mean /= (double)n;
    for (int i = 0; i < n; i++) {
        double d = t->d[i].adv - mean;
        var += d * d;
    }
    var /= (double)n;
    double sd = sqrt(var) + 1e-8;

    int *order = (int *)xmalloc(sizeof(int) * (size_t)n);
    for (int i = 0; i < n; i++) order[i] = i;

    uint64_t shuf[4];
    rng_seed(shuf, (uint64_t)n * 6364136223846793005ULL + 1442695040888963407ULL);

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        for (int i = n - 1; i > 0; i--) {
            int j = (int)(rng_next(shuf) % (uint64_t)(i + 1));
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
        for (int start = 0; start < n; start += cfg->minibatch) {
            int end = start + cfg->minibatch;
            if (end > n) end = n;
            int cnt = end - start;
            if (cnt < 1) continue;

            for (int k = start; k < end; k++) {
                Step *s = &t->d[order[k]];
                double adv = (s->adv - mean) / sd;
                net_forward(net, s->obs);
                /* Same mask as when the action was taken. An illegal action
                 * then has probability 0, so both terms of its gradient below
                 * vanish and the entropy is taken over the legal ones only. */
                net_mask(net, s->mask);
                double p_act = net->probs[s->action];
                double logp = log(p_act > 1e-12 ? p_act : 1e-12);
                double ent = net_entropy(net);

                double dlogp;
                if (use_clip) {
                    double ratio = exp(logp - s->logp_old);
                    double surr1 = ratio * adv;
                    double clipped = fmin(fmax(ratio, 1.0 - cfg->clip), 1.0 + cfg->clip);
                    double surr2 = clipped * adv;
                    dlogp = (surr1 <= surr2) ? (-adv * ratio) : 0.0;
                } else {
                    dlogp = -adv;
                }

                double dlogits[MAX_ACT];
                for (int a = 0; a < net->nact; a++) {
                    double onehot = (a == s->action) ? 1.0 : 0.0;
                    double g = dlogp * (onehot - net->probs[a]);
                    g += cfg->entropy * net->probs[a] * (log(net->probs[a] + 1e-12) + ent);
                    dlogits[a] = g;
                }
                double dv = cfg->vf_coef * (net->value - s->ret);
                net_backward(net, dlogits, dv);
            }
            net_clip_grads(net, 1.0);
            double scale = 1.0 / (double)cnt;
            for (int i = 0; i < net->total; i++) net->g[i] *= scale;
            net_adam(net, cfg->lr, 1e-5);
        }
    }
    free(order);
}

/* ------------------------------------------------------------------ */
/* Rewind-and-fork: exact counterfactual action values                 */
/* ------------------------------------------------------------------ */

/* Roll forward from `snap`, forcing `action` on the first decision of `agent`
 * in the first step. Returns the discounted reward for that agent.
 * Every step consumed counts against the global training budget. */
/* Roll a branch from `snap`, forcing `action` at the first decision.

   `harvest` (optional) receives the branch's own decisions. Every step after
   the forced one is drawn from the current policy, so those transitions are
   ordinary on-policy experience -- they can go straight into the same batch as
   the main rollout. That is what stops a branch from being pure measurement.

   Returns the discounted return for `agent`. Every step counts against the
   training budget.

   With a legal-move mask the forced action can turn out to be illegal at the
   decision the branch actually reaches. Then vm->force_missed is set on
   return and the value means nothing. */
static double fork_rollout(VM *vm, const State *snap, int agent, int action,
                           Traj *harvest, int ep_id) {
    State *saved_s = vm->s;
    int saved_record = vm->record;
    int saved_rec_agent = vm->record_agent;
    int saved_det = vm->deterministic;
    int saved_force = vm->force_agent;
    int saved_force_act = vm->force_action;

    vm->branch = *snap;
    vm->s = &vm->branch;
    /* Record only this agent's decisions so the terminal value is its own. */
    vm->record = 1;
    vm->record_agent = agent;
    vm->deterministic = 0;
    vm->force_agent = agent;
    vm->force_action = action;
    vm->force_missed = 0;

    const Program *p = vm->p;
    /* A branch must not invent ticks past the end of the episode: the taken
     * action's value comes from the real return, so a branch that keeps
     * simulating after the episode ended is not comparable with it. */
    int left = p->horizon - (int)snap->step;
    if (left < 1) left = 1;
    int horizon = left < vm->fork_horizon ? left : vm->fork_horizon;
    double total = 0.0, disc = 1.0, last_v = 0.0;
    int have_v = 0, terminal = 0;

    for (int t = 0; t < horizon; t++) {
        if (vm->branch.done) { terminal = 1; break; }
        if (!vm_step(vm)) { terminal = 1; break; }
        /* The action could not be played where the branch reached the decision
         * (see OP_HUNCH), so this is not a measurement of it. Stop here; the
         * caller reads vm->force_missed and discards the result. */
        if (vm->force_missed) break;
        total += disc * vm->agent_reward[agent];
        disc *= vm->gamma;
        if (vm->branch.done) { terminal = 1; break; }
        if (vm->n_dec > 0) { last_v = vm->dec[0].value; have_v = 1; }
        else have_v = 0;

        if (harvest && vm->n_dec > 0 && harvest->n < harvest->cap) {
            Decision *d = &vm->dec[vm->n_dec - 1];
            Step *st = &harvest->d[harvest->n];
            memset(st, 0, sizeof(*st));
            st->agent = agent;
            st->ep = ep_id;
            st->action = d->action;
            st->logp = d->logp;
            st->logp_old = d->logp;
            st->value = d->value;
            st->reward = vm->agent_reward[agent];
            memcpy(st->obs, d->obs, sizeof(double) * (size_t)d->obs_dim);
            memcpy(st->prob, d->prob, sizeof(double) * (size_t)d->n_act);
            st->obs_dim = d->obs_dim;
            st->n_act = d->n_act;
            st->mask = d->mask;
            st->valid = 1;
            harvest->n++;
        }
    }
    /* Tail of the episode is carried by the critic instead of being replayed. */
    if (!terminal && have_v) total += disc * last_v;

    vm->s = saved_s;
    vm->record = saved_record;
    vm->record_agent = saved_rec_agent;
    vm->deterministic = saved_det;
    vm->force_agent = saved_force;
    vm->force_action = saved_force_act;
    return total;
}

typedef struct { double score; int idx; } Scored;

static int cmp_scored_desc(const void *a, const void *b) {
    double x = ((const Scored *)a)->score;
    double y = ((const Scored *)b)->score;
    if (x < y) return 1;
    if (x > y) return -1;
    return 0;
}

/* Apply exact counterfactual advantages for one forked state.
 * Q[a] is the measured value of taking action a at that instant. */
static void fork_apply(Net *net, const Step *st, const double *Q, int n_act,
                       const Cfg *cfg, double *acc_g, int *acc_n, double *value_target,
                       int taken) {
    net_forward(net, st->obs);
    net_mask(net, st->mask);
    /* An illegal action was never rolled and has no value, so it is left out
     * of the baseline, the mean and the spread. */
    uint32_t legal = st->mask;
    int n_legal = mask_count(legal, n_act);
    double base = 0.0;
    for (int a = 0; a < n_act; a++)
        if ((legal >> a) & 1u) base += net->probs[a] * Q[a];

    double qmean = 0.0, qvar = 0.0;
    for (int a = 0; a < n_act; a++)
        if ((legal >> a) & 1u) qmean += Q[a];
    qmean /= (double)n_legal;
    for (int a = 0; a < n_act; a++) {
        if (!((legal >> a) & 1u)) continue;
        double d = Q[a] - qmean;
        qvar += d * d;
    }
    /* Floor the scale relative to the size of the values: when every action is
     * worth about the same, a tiny spread is noise, not signal. */
    double qsd = sqrt(qvar / (double)n_legal);
    double floor_sd = 0.03 * (fabs(base) + 1.0);
    if (qsd < floor_sd) qsd = floor_sd;

    if (cfg->fork_inject) {
        /* Hand the exact advantage back to the main trajectory. There is then a
         * single, coherent policy update instead of two competing ones. */
        const Step *cs = st;
        if (taken >= 0 && taken < n_act) {
            double exact = (Q[taken] - base) / qsd;
            if (exact > 5.0) exact = 5.0;
            if (exact < -5.0) exact = -5.0;
            ((Step *)cs)->adv = exact;
        }
        double dv = cfg->vf_coef * (net->value - base);
        double zero[MAX_ACT];
        for (int a = 0; a < MAX_ACT; a++) zero[a] = 0.0;
        net_backward(net, zero, dv);
        *acc_n += 1;
        *value_target += base;
        return;
    }

    double ent = net_entropy(net);
    double dlogits[MAX_ACT];
    for (int a = 0; a < n_act; a++) {
        if (!((legal >> a) & 1u)) { dlogits[a] = 0.0; continue; }
        double adv = (Q[a] - base) / qsd;
        if (adv > 5.0) adv = 5.0;
        if (adv < -5.0) adv = -5.0;
        /* dJ/dz[k] = p[k] * A[k]  (A is policy-centred, so the sum term vanishes) */
        double g = cfg->fork_pg ? -net->probs[a] * adv : 0.0;
        g += cfg->entropy * net->probs[a] * (log(net->probs[a] + 1e-12) + ent);
        dlogits[a] = g;
    }
    double dv = cfg->vf_coef * (net->value - base);
    net_backward(net, dlogits, dv);
    *acc_n += 1;
    *value_target += base;
}

static long fork_pass(VM *vm, Net *net, Traj *t, SnapLog *sl, const Cfg *cfg,
                      long fork_budget, int agent_idx, int harvest_ep0) {
    if (sl->n == 0 || fork_budget <= 0 || t->n == 0) return 0;

    Scored *cand = (Scored *)xmalloc(sizeof(Scored) * (size_t)sl->n);
    int ncand = 0;
    for (int i = 0; i < sl->n; i++) {
        if (sl->agent[i] != agent_idx) continue;
        int ti = sl->traj_idx[i];
        if (ti < 0 || ti >= t->n) continue;
        Step *st = &t->d[ti];
        /* With one legal action there is no alternative to measure. */
        if (mask_count(st->mask, st->n_act) < 2) continue;
        double ent = 0.0;
        for (int a = 0; a < st->n_act; a++) {
            double p = st->prob[a];
            if (p > 1e-12) ent -= p * log(p);
        }
        double surprise = fabs(st->reward - st->value);
        /* An early decision has more of the episode downstream of it, and is
         * cheaper to bootstrap from. Nudge those to the front of the queue. */
        double leverage = 1.0 - (double)sl->step_idx[i] /
            (double)(vm->p->horizon > 0 ? vm->p->horizon : 1);
        cand[ncand].score = ent + surprise + 0.5 * leverage;
        cand[ncand].idx = i;
        ncand++;
    }
    if (ncand == 0) { free(cand); return 0; }
    qsort(cand, (size_t)ncand, sizeof(Scored), cmp_scored_desc);

    const Program *p = vm->p;
    long used = 0;
    int applied = 0;
    int skipped = 0;    /* fork points dropped because a branch missed */
    double acc_g = 0.0;
    int acc_n = 0;
    double vtarget = 0.0;

    for (int c = 0; c < ncand && applied < cfg->fork_max_points; c++) {
        int i = cand[c].idx;
        int ti = sl->traj_idx[i];
        Step *st = &t->d[ti];
        int n_act = st->n_act;
        if (n_act > net->nact) n_act = net->nact;

        long remaining = p->horizon - sl->step_idx[i];
        if (remaining > vm->fork_horizon) remaining = vm->fork_horizon;
        long est = (long)mask_count(st->mask, n_act) * remaining;
        if (used + est > fork_budget && applied + skipped > 0) break;

        double Q[MAX_ACT];
        long before = vm->total_steps;
        int ok = 1, missed = 0;
        for (int a = 0; a < n_act; a++) {
            if (!((st->mask >> a) & 1u)) {
                /* Not an alternative: never rolled, and fork_apply skips it. */
                Q[a] = 0.0;
                continue;
            }
            if (a == sl->action[i]) {
                // free and exact: the real continuation from this snapshot
                Q[a] = st->rtg;
                continue;
            }
            Q[a] = fork_rollout(vm, &sl->snap[i], sl->agent[i], a,
                                cfg->harvest ? t : NULL,
                                harvest_ep0 + applied * MAX_ACT + a);
            if (vm->total_steps >= vm->max_steps) { ok = 0; }
            if (vm->force_missed) { missed = 1; break; }
        }
        used += vm->total_steps - before;
        if (!ok) break;
        /* One alternative could not be played in the replay, so this state has
         * no complete set of values. Leave its sampled advantage alone. */
        if (missed) { skipped++; continue; }

        fork_apply(net, st, Q, n_act, cfg, &acc_g, &acc_n, &vtarget, st->action);
        applied++;

        if (vm->total_steps >= vm->max_steps) break;
    }

    if (acc_n > 0) {
        net_clip_grads(net, 1.0);
        double scale = 1.0 / (double)acc_n;
        for (int i = 0; i < net->total; i++) net->g[i] *= scale;
        /* A gentler step than the PPO pass: this is an extra correction on top
         * of an already-updated policy, not the main update. */
        net_adam(net, cfg->lr * 0.5, 1e-5);
    }
    free(cand);
    return used;
}

/* ------------------------------------------------------------------ */
/* Evaluation                                                          */
/* ------------------------------------------------------------------ */

static double eval_policy(const Program *p, Net **nets, int episodes, uint64_t seed,
                          int horizon, int agent) {
    State s;
    VM vm;
    vm_init(&vm, p, &s);
    for (int i = 0; i < p->nagents; i++) vm.nets[i] = nets ? nets[i] : NULL;
    vm.max_steps = 0;
    vm.deterministic = 0;

    double total = 0.0;
    double ret[MAX_AGENTS];
    for (int e = 0; e < episodes; e++) {
        run_episode(&vm, NULL, NULL, seed + (uint64_t)e * 7919ULL, 0, e, ret, horizon);
        total += ret[agent];
    }
    return total / (double)episodes;
}

static double eval_policy_frozen(const Program *p, Net **nets, int episodes, uint64_t seed,
                                 int horizon, int agent, int fz_agent, int fz_action) {
    State s;
    VM vm;
    vm_init(&vm, p, &s);
    for (int i = 0; i < p->nagents; i++) vm.nets[i] = nets ? nets[i] : NULL;
    vm.max_steps = 0;
    vm.deterministic = 0;
    vm.freeze_agent = fz_agent;
    vm.freeze_action = fz_action;

    double total = 0.0;
    double ret[MAX_AGENTS];
    for (int e = 0; e < episodes; e++) {
        run_episode(&vm, NULL, NULL, seed + (uint64_t)e * 7919ULL, 0, e, ret, horizon);
        total += ret[agent];
    }
    return total / (double)episodes;
}

/* ------------------------------------------------------------------ */
/* Policy IO                                                           */
/* ------------------------------------------------------------------ */

static void save_policy(const char *path, const Program *p, Net **nets) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "hunch-vm: cannot write %s: %s\n", path, strerror(errno));
        exit(1);
    }
    fprintf(f, "HUNCHPOL 1\nworld %s\nagents %d\n", p->world, p->nagents);
    for (int i = 0; i < p->nagents; i++) {
        Net *n = nets[i];
        fprintf(f, "agent %d %s %d %d %d\n", i, p->agents[i].name, n->obs, n->nact, n->total);
        for (int k = 0; k < n->total; k++)
            fprintf(f, "%.17g\n", n->w[k]);
    }
    fclose(f);
}

static int load_policy(const char *path, const Program *p, Net **nets) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char tag[64];
    int version = 0;
    if (fscanf(f, "%63s %d", tag, &version) != 2) { fclose(f); return 0; }
    if (strcmp(tag, "HUNCHPOL") != 0) { fclose(f); return 0; }
    fscanf(f, "%63s %63s", tag, tag);
    int nagents = 0;
    if (fscanf(f, "%63s %d", tag, &nagents) != 2) { fclose(f); return 0; }
    for (int i = 0; i < nagents; i++) {
        int idx, obs, nact, total;
        char name[48];
        if (fscanf(f, "%63s %d %47s %d %d %d", tag, &idx, name, &obs, &nact, &total) != 6) {
            fclose(f);
            return 0;
        }
        if (idx < 0 || idx >= p->nagents || idx >= MAX_AGENTS) { fclose(f); return 0; }
        Net *n = nets[idx];
        if (!n || n->total != total || n->obs != obs || n->nact != nact) { fclose(f); return 0; }
        for (int k = 0; k < total; k++) {
            if (fscanf(f, "%lf", &n->w[k]) != 1) { fclose(f); return 0; }
        }
    }
    fclose(f);
    return 1;
}

/* ------------------------------------------------------------------ */
/* JSON helpers                                                        */
/* ------------------------------------------------------------------ */

static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (const char *c = s; *c; c++) {
        if (*c == '"' || *c == '\\') { fputc('\\', f); fputc(*c, f); }
        else if (*c == '\n') fputs("\\n", f);
        else fputc(*c, f);
    }
    fputc('"', f);
}

static void json_num(FILE *f, double v) {
    if (!(v == v) || v > 1e300 || v < -1e300) { fputs("0", f); return; }
    if (v == (double)(long long)v && fabs(v) < 1e15) fprintf(f, "%lld", (long long)v);
    else fprintf(f, "%.6g", v);
}

/* ------------------------------------------------------------------ */
/* Training                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    double step;
    double ret;
} CurvePoint;

typedef struct {
    CurvePoint *pts;
    int n, cap;
} Curve;

/* --progress: say each checkpoint on stderr as it is reached. */
static int g_progress = 0;

static void curve_push(Curve *c, double step, double ret) {
    if (g_progress) {
        fprintf(stderr, "step %.0f %.9g\n", step, ret);
        fflush(stderr);
    }
    if (c->n >= c->cap) return;
    c->pts[c->n].step = step;
    c->pts[c->n].ret = ret;
    c->n++;
}

static double train(const Program *p, const char *algo, Cfg *cfg, uint64_t seed,
                    Net **nets, Curve *curve) {
    State s;
    VM vm;
    vm_init(&vm, p, &s);
    vm.gamma = cfg->gamma;
    vm.fork_horizon = cfg->fork_horizon > 0 ? cfg->fork_horizon : 24;
    vm.max_steps = cfg->budget;
    vm.deterministic = 0;

    int learn = strcmp(algo, "random") != 0;

    if (learn) {
        for (int i = 0; i < p->nagents; i++)
            nets[i] = net_new(p->agents[i].obs_dim, p->agents[i].n_act,
                              seed + (uint64_t)i * 1000003ULL);
    }
    for (int i = 0; i < p->nagents; i++) vm.nets[i] = nets[i];

    int horizon = cfg->horizon > 0 ? cfg->horizon : p->horizon;

    Traj traj[MAX_AGENTS];
    int traj_cap = cfg->rollout_steps * 4 + horizon * 8;
    batch_init(traj, p->nagents, traj_cap);
    SnapLog sl;
    int snapcap = cfg->rollout_steps + horizon * 8;
    snaplog_init(&sl, snapcap);

    uint64_t ep_seed = seed * 6364136223846793005ULL + 1;

    int n_checks = curve ? 12 : 0;
    int checks_done = 0;
    if (curve && n_checks > 0) {
        double r = eval_policy(p, nets, cfg->eval_episodes, EVAL_SEED, horizon, 0);
        curve_push(curve, 0, r);
        checks_done = 1;
    }

    double t0 = now_sec();

    while (vm.total_steps < cfg->budget) {
        batch_clear(traj, p->nagents);
        sl.n = 0;
        long iter_start = vm.total_steps;
        long rollout_target = cfg->rollout_steps;

        while (vm.total_steps - iter_start < rollout_target && vm.total_steps < cfg->budget) {
            ep_seed = ep_seed * 6364136223846793005ULL + 1442695040888963407ULL;
            run_episode(&vm, learn ? traj : NULL, learn ? &sl : NULL,
                        ep_seed ^ seed, learn, (int)(ep_seed & 0x7fffffff), NULL, horizon);
        }
        long main_used = vm.total_steps - iter_start;
        if (!learn) continue;

        if (strcmp(algo, "reinforce") == 0) {
            for (int a = 0; a < p->nagents; a++) {
                compute_mc(&traj[a], cfg->gamma);
                policy_update(nets[a], &traj[a], cfg, 1, 0);
            }
        } else {
            long fork_used_total = 0;
            /* Forking buys credit assignment, and it is worth most while the
             * critic is still wrong. Spend it early, then get out of the way
             * and let the cheap on-policy updates finish the job. */
            double progress = cfg->budget > 0
                ? (double)vm.total_steps / (double)cfg->budget : 1.0;
            double anneal = 1.0;
            if (cfg->fork_decay > 0) {
                anneal = 1.0 - progress / cfg->fork_decay;
                if (anneal < 0) anneal = 0;
                if (anneal > 1) anneal = 1;
            }
            long fork_budget = (long)((double)main_used * cfg->fork_ratio * anneal);
            int is_fork = strcmp(algo, "fork") == 0 && fork_budget > 0;

            if (is_fork) {
                /* Returns-to-go on the main rollout are needed first: they give
                 * the value of the action actually taken at no extra cost. */
                for (int a = 0; a < p->nagents; a++) {
                    compute_rtg(&traj[a], cfg->gamma);
                    if (cfg->fork_inject)
                        compute_gae(&traj[a], cfg->gamma, cfg->lambda);
                }
                int harvest_ep = 1000000;
                for (int a = 0; a < p->nagents; a++) {
                    if (vm.total_steps >= cfg->budget) break;
                    if (traj[a].n == 0) continue;
                    fork_used_total += fork_pass(&vm, nets[a], &traj[a], &sl, cfg,
                                                 fork_budget / p->nagents, a,
                                                 harvest_ep);
                    harvest_ep += 1000000;
                }
            }
            (void)fork_used_total;

            for (int a = 0; a < p->nagents; a++) {
                if (!is_fork || !cfg->fork_inject || cfg->harvest)
                    compute_gae(&traj[a], cfg->gamma, cfg->lambda);
                policy_update(nets[a], &traj[a], cfg, cfg->ppo_epochs, 1);
            }
        }

        if (curve && checks_done < n_checks) {
            double frac = (double)vm.total_steps / (double)cfg->budget;
            double next = (double)checks_done / (double)n_checks;
            if (frac >= next) {
                double r = eval_policy(p, nets, cfg->eval_episodes, EVAL_SEED,
                                       horizon, 0);
                curve_push(curve, (double)vm.total_steps, r);
                checks_done++;
            }
        }
    }

    double wall = now_sec() - t0;
    (void)wall;
    if (curve) {
        double r = eval_policy(p, nets, cfg->eval_episodes, EVAL_SEED, horizon, 0);
        curve_push(curve, (double)vm.total_steps, r);
    }

    batch_free(traj, p->nagents);
    snaplog_free(&sl);
    return (double)vm.total_steps;
}

/* ------------------------------------------------------------------ */
/* Trace output (for the Studio)                                       */
/* ------------------------------------------------------------------ */

static void emit_state(FILE *f, const Program *p, const State *s) {
    fputc('[', f);
    for (int i = 0; i < p->nslots; i++) {
        if (i) fputc(',', f);
        json_num(f, s->v[i]);
    }
    fputc(']', f);
}

static void emit_meta(FILE *f, const Program *p) {
    fputs("{\"world\":", f);
    json_str(f, p->world);
    fprintf(f, ",\"horizon\":%d,\"agents\":[", p->horizon);
    for (int i = 0; i < p->nagents; i++) {
        if (i) fputc(',', f);
        fputs("{\"name\":", f);
        json_str(f, p->agents[i].name);
        fprintf(f, ",\"obs_dim\":%d,\"n_act\":%d}", p->agents[i].obs_dim,
                p->agents[i].n_act);
    }
    fputs("],\"fields\":[", f);
    for (int i = 0; i < p->ndecls; i++) {
        if (i) fputc(',', f);
        fputs("{\"name\":", f);
        json_str(f, p->decls[i].name);
        fprintf(f, ",\"base\":%d,\"len\":%d,\"is_int\":%d}", p->decls[i].base,
                p->decls[i].len, p->decls[i].is_int);
    }
    fputs("],\"nslots\":", f);
    fprintf(f, "%d", p->nslots);
    fputs(",\"episodes\":[", f);
}

typedef struct {
    double q[MAX_ACT];
    int n_act;
    int agent;
    int step;
    double prob[MAX_ACT];
    double state_snapshot;
    int has_q;
} ForkRec;

/* Run a single traced episode with an optional policy, optionally measuring
 * exact counterfactual action values at each decision. */
static void trace_episode(const Program *p, Net **nets, uint64_t seed, int horizon,
                          FILE *out, int want_forks, VM *vm, State *s) {
    state_reset(p, s, seed);
    s->ep = 0;
    g_trace_on = 1;
    g_trace_seed = seed;
    for (int i = 0; i < p->nagents; i++) vm->nets[i] = nets ? nets[i] : NULL;
    vm->record = 1;
    vm->deterministic = 0;
    vm->force_agent = -1;
    vm->max_steps = 0;
    vm_run(vm, p->init_entry);

    fputs("{\"seed\":", out);
    fprintf(out, "%llu", (unsigned long long)seed);
    fputs(",\"start\":", out);
    emit_state(out, p, s);
    fputs(",\"steps\":[", out);

    int first = 1, awaiting = 0;
    for (int t = 0; t < horizon; t++) {
        if (s->done) break;
        State snap = *s;
        vm->record = 1;
        if (!vm_step(vm)) break;
        vm->record = 0;
        int asking = vm->stopped == 3;

        if (!first) fputc(',', out);
        first = 0;

        fputs("{\"s\":", out);
        emit_state(out, p, s);
        fputs(",\"a\":{", out);
        for (int k = 0; k < vm->n_dec; k++) {
            if (k) fputc(',', out);
            json_str(out, p->agents[vm->dec[k].agent].name);
            fprintf(out, ":%d", vm->dec[k].action);
        }
        fputs("},\"r\":{", out);
        for (int a = 0; a < p->nagents; a++) {
            if (a) fputc(',', out);
            json_str(out, p->agents[a].name);
            fputc(':', out);
            json_num(out, vm->agent_reward[a]);
        }
        fputs("},\"p\":{", out);
        for (int k = 0; k < vm->n_dec; k++) {
            if (k) fputc(',', out);
            json_str(out, p->agents[vm->dec[k].agent].name);
            fputs(":[", out);
            for (int a = 0; a < vm->dec[k].n_act; a++) {
                if (a) fputc(',', out);
                json_num(out, vm->dec[k].prob[a]);
            }
            fputc(']', out);
        }
        fputs("}", out);
        /* Every decision of the step in the order it was made, with what the
         * agent was shown. The maps above keep one entry per agent. */
        fputs(",\"d\":[", out);
        for (int k = 0; k < vm->n_dec; k++) {
            const Decision *d = &vm->dec[k];
            fprintf(out, "%s{\"agent\":%d,\"action\":%d,\"legal\":%u,\"obs\":[", k ? "," : "",
                    d->agent, d->action, d->mask);
            for (int i = 0; i < d->obs_dim; i++) {
                if (i) fputc(',', out);
                json_num(out, d->obs[i]);
            }
            fputs("],\"probs\":[", out);
            for (int a = 0; a < d->n_act; a++) {
                if (a) fputc(',', out);
                json_num(out, d->prob[a]);
            }
            fputs("]", out);
            if (d->searched) {
                /* chosen by a search: how many playouts tried each action */
                fputs(",\"tries\":[", out);
                for (int a = 0; a < d->n_act; a++) fprintf(out, "%s%d", a ? "," : "", d->tries[a]);
                fputs("]", out);
            }
            fputs("}", out);
        }
        fputs("]", out);
        if (asking) {
            /* The step stopped part way, at a decision of the scripted agent. */
            fputs(",\"partial\":true}", out);
            awaiting = 1;
            break;
        }

        if (want_forks && nets) {
            /* Kept now: the branches below reuse vm->dec. */
            Decision found[MAX_DEC_PER_STEP];
            int searched[MAX_DEC_PER_STEP];
            for (int k = 0; k < vm->n_dec; k++) {
                searched[k] = vm->dec[k].searched;
                if (searched[k]) found[k] = vm->dec[k];
            }
            fputs(",\"q\":{", out);
            int wrote = 0;
            for (int k = 0; k < vm->n_dec; k++) {
                int agent = vm->dec[k].agent;
                int n_act = vm->dec[k].n_act;
                /* Read now: the branches below reuse vm->dec. */
                uint32_t legal = vm->dec[k].mask;
                if (n_act > 8) n_act = 8;
                if (wrote++) fputc(',', out);
                json_str(out, p->agents[agent].name);
                fputs(":[", out);
                for (int a = 0; a < n_act; a++) {
                    if (a) fputc(',', out);
                    /* An illegal action has no value to measure; null keeps
                     * the array aligned with the action numbers. */
                    if (!((legal >> a) & 1u)) { fputs("null", out); continue; }
                    if (searched[k]) {
                        /* A search has measured this already: the mean of
                         * what the playouts that tried the action saw.
                         * Replaying the episode with a search in every seat
                         * would take minutes for each number. */
                        if (found[k].tries[a] > 0) json_num(out, found[k].mean[a]);
                        else fputs("null", out);
                        continue;
                    }
                    double q = fork_rollout(vm, &snap, agent, a, NULL, 0);
                    if (vm->force_missed) fputs("null", out);
                    else json_num(out, q);
                }
                fputc(']', out);
            }
            fputs("}", out);
        }
        fputs("}", out);
    }
    fputs("]", out);
    if (awaiting) {
        fprintf(out, ",\"awaiting\":{\"agent\":%d,\"step\":%lld,\"n\":%d,\"legal\":%u,\"obs\":[",
                vm->ask_agent, (long long)s->step, vm->ask_n, vm->ask_legal);
        for (int i = 0; i < vm->ask_obs_len; i++) {
            if (i) fputc(',', out);
            json_num(out, vm->scratch[i]);
        }
        fputs("]}", out);
    }
    fputs("}", out);
}

/* ------------------------------------------------------------------ */
/* Argument parsing                                                    */
/* ------------------------------------------------------------------ */

static const char *arg_value(int argc, char **argv, const char *flag, const char *dflt) {
    for (int i = 2; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return dflt;
}

static long arg_long(int argc, char **argv, const char *flag, long dflt) {
    const char *v = arg_value(argc, argv, flag, NULL);
    return v ? atol(v) : dflt;
}

static double arg_double(int argc, char **argv, const char *flag, double dflt) {
    const char *v = arg_value(argc, argv, flag, NULL);
    return v ? atof(v) : dflt;
}

static int has_flag(int argc, char **argv, const char *flag) {
    for (int i = 2; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* The network a search learns to lean on                              */
/* ------------------------------------------------------------------ */

/* A search can be guided by what its earlier games taught it, in the manner
 * of AlphaZero. A small network reads what the agent to move is shown and
 * says two things: which actions are worth trying first (learned from where
 * earlier searches spent their playouts) and what every agent goes on to
 * collect from here (learned from how those games ended). A guided search
 * leaves its tree at the first position it has not met, as before, and
 * takes as that position's value a mix: `rollout` (half) of one random
 * playout from it, and the rest the network's estimate, as AlphaGo did. On
 * Connect Four the network's value alone played worse than the mix, and on
 * a 5 by 4 board random playouts alone did.
 *
 * What the agent is shown is turned into inputs part by part. A part that
 * took at most GUIDE_ONEHOT different values while the world was played at
 * random (a board cell: empty, mine, theirs) gets one input per value, 1 for
 * the value it has; any other part is one input, centred and scaled. When
 * every agent is shown the same shape and has the same actions, they share
 * one network, with one more input per agent saying whose turn it is. */
#define GUIDE_ONEHOT  8
#define GUIDE_MAX_HID 256

typedef struct {
    int obs_dim, nact, nval, hid, shared, nin;
    int nv[MAX_OBS];                    /* values with an input each; 0: one scaled input */
    double val[MAX_OBS][GUIDE_ONEHOT];
    double mean[MAX_OBS], scale[MAX_OBS];
    int off[MAX_OBS];                   /* first input of each part */
    int o_w1, o_b1, o_w2, o_b2, o_wp, o_bp, o_wv, o_bv, total;
    double *w, *g, *m, *v;              /* g, m and v only while training */
    long opt_step;
} GNet;

typedef struct {
    int nnets;
    int net_of[MAX_AGENTS];
    GNet *net[MAX_AGENTS];
    double vscale;                      /* values are learned divided by this */
    /* how it was made, for the report */
    long games, samples, rounds;
    double sec;
    int cores, playouts;
} Guide;

/* One pass through a network, kept for its gradient. */
typedef struct {
    int n;                              /* inputs that are not zero */
    int idx[MAX_OBS + MAX_AGENTS];
    double x[MAX_OBS + MAX_AGENTS];
    double h1[GUIDE_MAX_HID], h2[GUIDE_MAX_HID];
    double probs[MAX_ACT], value[MAX_AGENTS];
} GPass;

static void gnet_layout(GNet *n) {
    int k = 0;
    for (int d = 0; d < n->obs_dim; d++) { n->off[d] = k; k += n->nv[d] ? n->nv[d] : 1; }
    n->nin = k + (n->shared ? n->nval : 0);
    int o = 0, H = n->hid;
    n->o_w1 = o; o += n->nin * H;       /* by input: an input's weights are side by side */
    n->o_b1 = o; o += H;
    n->o_w2 = o; o += H * H;
    n->o_b2 = o; o += H;
    n->o_wp = o; o += n->nact * H;
    n->o_bp = o; o += n->nact;
    n->o_wv = o; o += n->nval * H;
    n->o_bv = o; o += n->nval;
    n->total = o;
}

static void gnet_free(GNet *n) {
    if (!n) return;
    free(n->w); free(n->g); free(n->m); free(n->v); free(n);
}

static void guide_free(Guide *G) {
    if (!G) return;
    for (int k = 0; k < G->nnets; k++) gnet_free(G->net[k]);
    free(G);
}

static void gnet_encode(const GNet *n, int agent, const double *obs, int len, GPass *ps) {
    ps->n = 0;
    for (int d = 0; d < n->obs_dim; d++) {
        double v = d < len ? obs[d] : 0.0;
        if (n->nv[d]) {
            for (int k = 0; k < n->nv[d]; k++)
                if (n->val[d][k] == v) {
                    ps->idx[ps->n] = n->off[d] + k;
                    ps->x[ps->n++] = 1.0;
                    break;
                }
        } else {
            ps->idx[ps->n] = n->off[d];
            ps->x[ps->n++] = (v - n->mean[d]) / n->scale[d];
        }
    }
    if (n->shared) {
        ps->idx[ps->n] = n->nin - n->nval + agent;
        ps->x[ps->n++] = 1.0;
    }
}

/* Two layers of rectified units; the policy over the legal actions and a
 * value for every agent. */
static void gnet_forward(const GNet *n, GPass *ps, int n_act, uint32_t legal) {
    const int H = n->hid;
    const double *w = n->w;
    double pre[GUIDE_MAX_HID];
    memcpy(pre, w + n->o_b1, sizeof(double) * (size_t)H);
    for (int k = 0; k < ps->n; k++) {
        const double *col = w + n->o_w1 + (size_t)ps->idx[k] * H;
        double xv = ps->x[k];
        for (int i = 0; i < H; i++) pre[i] += xv * col[i];
    }
    for (int i = 0; i < H; i++) ps->h1[i] = pre[i] > 0 ? pre[i] : 0.0;
    for (int i = 0; i < H; i++) {
        const double *row = w + n->o_w2 + (size_t)i * H;
        double acc = w[n->o_b2 + i];
        for (int j = 0; j < H; j++) acc += row[j] * ps->h1[j];
        ps->h2[i] = acc > 0 ? acc : 0.0;
    }
    double maxl = -HUGE_VAL, sum = 0.0;
    for (int a = 0; a < n_act; a++) {
        ps->probs[a] = 0.0;
        if (!((legal >> a) & 1u)) continue;
        const double *row = w + n->o_wp + (size_t)a * H;
        double acc = w[n->o_bp + a];
        for (int i = 0; i < H; i++) acc += row[i] * ps->h2[i];
        ps->probs[a] = acc;
        if (acc > maxl) maxl = acc;
    }
    for (int a = 0; a < n_act; a++)
        if ((legal >> a) & 1u) { ps->probs[a] = exp(ps->probs[a] - maxl); sum += ps->probs[a]; }
    for (int a = 0; a < n_act; a++) ps->probs[a] /= sum;
    for (int j = 0; j < n->nval; j++) {
        const double *row = w + n->o_wv + (size_t)j * H;
        double acc = w[n->o_bv + j];
        for (int i = 0; i < H; i++) acc += row[i] * ps->h2[i];
        ps->value[j] = acc;
    }
}

/* What the network says at a decision of `agent`: ps->probs over the legal
 * actions, and ps->value in the world's own units. */
static void guide_eval(const Guide *G, int agent, const double *obs, int len, int n_act,
                       uint32_t legal, GPass *ps) {
    const GNet *n = G->net[G->net_of[agent]];
    gnet_encode(n, agent, obs, len, ps);
    gnet_forward(n, ps, n_act, legal);
    for (int j = 0; j < n->nval; j++) ps->value[j] *= G->vscale;
}

/* ------------------------------------------------------------------ */
/* Search: a policy that forks the world at each decision              */
/* ------------------------------------------------------------------ */

/* A world too big to follow whole is searched where it stands. At a decision
 * the instant is copied into the search's own VM and played on from there
 * many times (Monte Carlo tree search, UCT): the first decisions of each
 * playout are chosen by what earlier playouts found, the rest uniformly at
 * random, and the action tried most is played. Each agent is taken to want
 * its own return, so this serves one agent, two, or more.
 *
 * The copy is the running block itself, stopped after the decision: state,
 * stack, calls and instruction counter. The search's VM continues it with an
 * action pushed, so the VM that asked is never touched, and no block is run
 * twice.
 *
 * It reads the whole state. Where an agent is shown less than that, the
 * search knows more than its agent should; `hidden` records what `train`
 * found out about that.
 *
 * It is deterministic. Its random numbers come from a stream started from
 * the instant searched (random stream included), and the world's own stream
 * is not drawn from, so an episode replays exactly. */

#define SEARCH_MAX_NODES 1000000L   /* positions one search remembers; playouts past that add none */

enum { HIDDEN_NOTHING = 0, HIDDEN_FOUND, HIDDEN_UNKNOWN };

typedef struct {
    uint64_t h1, h2;    /* the instant, as two independent hashes */
    int agent, n_act;
    uint32_t legal;
    int visits;
    long stat;          /* its first entry in `tries` and `total` */
} SNode;

typedef struct { long node; int action; double before; } SPath;

/* A decision of a game the search played against itself, for the network to
 * learn from: what the agent was shown, where the search spent its playouts,
 * and what each agent went on to collect. */
typedef struct {
    long game;
    int ply, agent, n_act, obs_len;
    uint32_t legal;
    float pi[MAX_ACT];
    double z[MAX_AGENTS];       /* collected before it, until the game ends */
    double obs[MAX_OBS];
} GSample;

typedef struct { GSample *d; long n, cap; } GBuf;

typedef struct Search {
    Driver dr;          /* first, so a Driver * is a Search * */
    const Program *p;
    int playouts;       /* per decision */
    double explore;     /* how readily a less-tried action is tried again */
    int hidden;         /* HIDDEN_*: what train's check said of this world */
    int horizon;
    VM vm;              /* its own: the VM that asked is left as it was */
    State st, root;
    double root_stack[STACK_SIZE], root_scratch[MAX_SCRATCH];
    int root_calls[CALL_DEPTH];
    SNode *nodes;
    long nn, cap;
    int width;          /* entries of `tries` and `total` per node */
    int *tries;
    double *total;
    long *slot, nslot;
    SPath *path;
    int npath, cappath;
    int rollout;        /* this playout has left the positions remembered */
    double paid[MAX_AGENTS];    /* in the blocks of this playout that have ended */
    double low, high;   /* the least and the most any decision has gone on to collect */
    uint64_t rng[4];
    /* what the last search found at its root */
    int found_n, found_tries[MAX_ACT];
    double found_mean[MAX_ACT];
    /* train measures how fast this world is searched: with `slice` set a
     * search stops after that many seconds. Never set in play, where the
     * count alone decides, so that play does not depend on the machine. */
    double slice;
    double cost;        /* seconds a playout took, summed over the searches timed */
    long played, searches;
    /* what train's check of the world saw (see shown_check), kept in the file */
    long checked[4];
    /* With a network (see "The network a search learns to lean on") the
     * value at the first position not met before is the network's, mixed
     * with one random playout from there: the playout's `leaf`, what each
     * agent collects by the end, counted from the start of the playout. */
    Guide *guide;
    float *prior;       /* per node, like `tries`: the network's policy there */
    double *v0;         /* per node: what the network said its agent collects from there */
    int leafed;
    double leaf[MAX_AGENTS];
    double mix;         /* the share of a leaf's value that is a random playout from it (rollout) */
    GPass pass;
    /* Self-play, while train makes the network: the root's policy gets noise,
     * the first `sample_moves` decisions of a game are drawn in proportion to
     * the playouts, and every decision is kept in `rec`. */
    int selfplay, sample_moves, moves;
    double noise;       /* the share of the root's policy that is noise */
    double ep_paid[MAX_AGENTS];         /* collected in this game before the step running */
    GBuf *rec;
    long game;
} Search;

/* The instant a VM stands at, asked a decision: who is asked, the step, where
 * in the block, the stack and the calls it is inside, and every slot of the
 * world's memory. What the block has paid so far is left out: it is the same
 * position whatever was collected on the way to it. */
static void search_key(const VM *vm, int agent, uint64_t *h1, uint64_t *h2) {
    uint64_t a = mix64(0x5EA4C4ULL + (uint64_t)agent, (uint64_t)vm->s->step);
    a = mix64(a, (uint64_t)vm->at_ip * 2 + (uint64_t)(vm->in_init != 0));
    a = mix64(a, (uint64_t)vm->sp * 1024 + (uint64_t)vm->csp);
    uint64_t b = a ^ 0xD1B54A32D192ED03ULL;
#define SEARCH_MIX(x) do { uint64_t x_ = (x); a = mix64(a, x_); \
        b = (rotl64(b, 23) ^ x_) * 0x9FB21C651E98DF25ULL; } while (0)
    for (int i = 0; i < vm->sp; i++) SEARCH_MIX(bits_of(vm->stack[i]));
    for (int i = 0; i < vm->csp; i++) SEARCH_MIX((uint64_t)vm->call_ip[i]);
    const double *v = vm->s->v;
    for (int i = 0, n = vm->p->nslots; i < n; i++) SEARCH_MIX(bits_of(v[i]));
#undef SEARCH_MIX
    *h1 = a;
    *h2 = b ^ (b >> 31);
}

static Search *search_new(const Program *p, int playouts, double explore, int hidden) {
    Search *S = (Search *)xcalloc(1, sizeof(Search));
    S->p = p;
    S->playouts = playouts < 1 ? 1 : playouts;
    S->explore = explore;
    S->hidden = hidden;
    S->horizon = p->horizon;
    return S;
}

static void search_free(Search *S) {
    if (!S) return;
    free(S->nodes); free(S->tries); free(S->total); free(S->slot); free(S->path);
    free(S->prior); free(S->v0);
    guide_free(S->guide);
    free(S);
}

/* The node for an instant, added if there is room. -1: no room, or a node
 * that does not fit what is asked here. */
static long search_node(Search *S, uint64_t h1, uint64_t h2, int agent, int n_act,
                        uint32_t legal, int *fresh) {
    long at = (long)(h1 & (uint64_t)(S->nslot - 1));
    *fresh = 0;
    for (;;) {
        long id = S->slot[at];
        if (id < 0) break;
        const SNode *n = &S->nodes[id];
        if (n->h1 == h1 && n->h2 == h2)
            return n->agent == agent && n->n_act == n_act && n->legal == legal ? id : -1;
        at = (at + 1) & (S->nslot - 1);
    }
    if (S->nn >= S->cap) return -1;
    long id = S->nn++;
    SNode *n = &S->nodes[id];
    n->h1 = h1; n->h2 = h2;
    n->agent = agent; n->n_act = n_act; n->legal = legal;
    n->visits = 0;
    n->stat = id * S->width;
    memset(S->tries + n->stat, 0, sizeof(int) * (size_t)n_act);
    memset(S->total + n->stat, 0, sizeof(double) * (size_t)n_act);
    S->slot[at] = id;
    *fresh = 1;
    return id;
}

/* UCB1 over a node's legal actions, with values scaled to the range seen so
 * far. An action not yet tried goes first, picked at random among those. */
static int search_pick(Search *S, long id) {
    const SNode *n = &S->nodes[id];
    const int *tries = S->tries + n->stat;
    const double *total = S->total + n->stat;
    if (S->guide) {
        /* PUCT, as in AlphaZero: the value found so far, scaled to the range
         * seen, plus a share of `explore` that the network's policy hands
         * out and that shrinks as an action is tried. An action not tried
         * yet is taken to be worth what the position is worth so far. */
        const float *prior = S->prior + n->stat;
        double sum = S->v0[id];
        for (int a = 0; a < n->n_act; a++) sum += total[a];
        double here = sum / (n->visits + 1.0);
        int known = S->high > S->low;
        double span = known ? S->high - S->low : 1.0;
        double push = S->explore * sqrt(n->visits + 1.0);
        int best = -1;
        double best_score = 0.0;
        for (int a = 0; a < n->n_act; a++) {
            if (!((n->legal >> a) & 1u)) continue;
            double q = tries[a] ? total[a] / tries[a] : here;
            double score = (known ? (q - S->low) / span : 0.5) + push * prior[a] / (1.0 + tries[a]);
            if (best < 0 || score > best_score) { best = a; best_score = score; }
        }
        return best;
    }
    int untried = 0;
    for (int a = 0; a < n->n_act; a++)
        if (((n->legal >> a) & 1u) && tries[a] == 0) untried++;
    if (untried) {
        int k = (int)(rng_double(S->rng) * untried);
        for (int a = 0; a < n->n_act; a++)
            if (((n->legal >> a) & 1u) && tries[a] == 0 && k-- == 0) return a;
    }
    double span = S->high > S->low ? S->high - S->low : 1.0;
    double log_n = log((double)n->visits + 1.0);
    int best = -1;
    double best_score = 0.0;
    for (int a = 0; a < n->n_act; a++) {
        if (!((n->legal >> a) & 1u)) continue;
        double score = (total[a] / tries[a] - S->low) / span +
                       S->explore * sqrt(log_n / tries[a]);
        if (best < 0 || score > best_score) { best = a; best_score = score; }
    }
    return best;
}

static void search_step_in(Search *S, long id, int action, double before) {
    if (S->npath >= S->cappath) {
        S->cappath = S->cappath ? S->cappath * 2 : 256;
        S->path = (SPath *)xrealloc(S->path, sizeof(SPath) * (size_t)S->cappath);
    }
    S->path[S->npath].node = id;
    S->path[S->npath].action = action;
    S->path[S->npath].before = before;
    S->npath++;
}

/* The network's word on a position: its policy kept with the node (unless
 * there is no room for one, id < 0), its values left in S->pass. */
static void search_expand(Search *S, long id, int agent, const double *obs, int obs_len,
                          int n_act, uint32_t legal) {
    guide_eval(S->guide, agent, obs, obs_len, n_act, legal, &S->pass);
    if (id < 0) return;
    const SNode *n = &S->nodes[id];
    for (int a = 0; a < n_act; a++) S->prior[n->stat + a] = (float)S->pass.probs[a];
    S->v0[id] = S->pass.value[agent];
}

/* Every decision of a playout after the first comes here. */
static int search_decide(Driver *dr, VM *vm, int agent, const double *obs, int obs_len,
                         int n_act, uint32_t legal, const double *probs) {
    Search *S = (Search *)dr;
    (void)probs;
    /* Past the remembered positions play is uniform, drawn by the VM from
     * the playout's own stream. */
    if (S->rollout) return DRIVE_PASS;
    uint64_t h1, h2;
    int fresh;
    search_key(vm, agent, &h1, &h2);
    long id = search_node(S, h1, h2, agent, n_act, legal, &fresh);
    if (S->guide && (id < 0 || fresh)) {
        /* A position not met before: the network says what it is worth, and
         * the playout ends here, or plays on at random to be mixed in. */
        search_expand(S, id, agent, obs, obs_len, n_act, legal);
        for (int a = 0; a < S->p->nagents; a++)
            S->leaf[a] = S->paid[a] + (vm->in_init ? 0.0 : vm->agent_reward[a]) + S->pass.value[a];
        S->leafed = 1;
        if (S->mix > 0) { S->rollout = 1; return DRIVE_PASS; }
        return DRIVE_STOP;
    }
    if (id < 0) { S->rollout = 1; return DRIVE_PASS; }
    int action = search_pick(S, id);
    /* What init pays is dropped by the runtime, so it is not counted here. */
    search_step_in(S, id, action, S->paid[agent] + (vm->in_init ? 0.0 : vm->agent_reward[agent]));
    if (fresh) S->rollout = 1;
    return action;
}

/* Called by a VM in the middle of a block, at a decision of `agent`: `ip` is
 * the instruction after it. Returns the action to play. */
static int search_choose(Search *S, VM *vm, int ip, int agent, int n_act, uint32_t legal,
                         const double *obs, int obs_len) {
    const Program *p = S->p;
    double t0 = S->slice > 0 ? now_sec() : 0.0;
    size_t state_bytes = offsetof(State, v) + sizeof(double) * (size_t)p->nslots;
    if (!S->nodes) {
        S->cap = (long)S->playouts + 1;
        if (S->cap > SEARCH_MAX_NODES) S->cap = SEARCH_MAX_NODES;
        for (int i = 0; i < p->nagents; i++)
            if (p->agents[i].n_act > S->width) S->width = p->agents[i].n_act;
        if (S->width < n_act) S->width = n_act;
        S->nslot = 64;
        while (S->nslot < S->cap * 2) S->nslot *= 2;
        S->nodes = (SNode *)xmalloc(sizeof(SNode) * (size_t)S->cap);
        S->tries = (int *)xmalloc(sizeof(int) * (size_t)S->cap * (size_t)S->width);
        S->total = (double *)xmalloc(sizeof(double) * (size_t)S->cap * (size_t)S->width);
        S->slot = (long *)xmalloc(sizeof(long) * (size_t)S->nslot);
        if (S->guide) {
            S->prior = (float *)xmalloc(sizeof(float) * (size_t)S->cap * (size_t)S->width);
            S->v0 = (double *)xmalloc(sizeof(double) * (size_t)S->cap);
        }
        vm_init(&S->vm, p, &S->st);
        S->dr.decide = search_decide;
    }
    if (n_act > S->width) fatal(2, "hunch: a decision with more actions than its agent declares");

    /* the instant: everything the rest of the block and the episode depend on */
    memcpy(&S->root, vm->s, state_bytes);
    memcpy(S->root_stack, vm->stack, sizeof(double) * (size_t)vm->sp);
    memcpy(S->root_calls, vm->call_ip, sizeof(int) * (size_t)vm->csp);
    memcpy(S->root_scratch, vm->scratch, sizeof(vm->scratch));
    int sp = vm->sp, csp = vm->csp, in_init = vm->in_init;
    long insns = vm->insns;

    S->nn = 0;
    for (long i = 0; i < S->nslot; i++) S->slot[i] = -1;
    S->low = HUGE_VAL;
    S->high = -HUGE_VAL;
    uint64_t h1, h2;
    int fresh;
    search_key(vm, agent, &h1, &h2);
    long root = search_node(S, h1, h2, agent, n_act, legal, &fresh);
    /* Its own stream, from the instant and the world's stream as it stands:
     * the same instant is searched the same way, wherever and whenever. */
    uint64_t seed = mix64(h1, h2);
    for (int i = 0; i < 4; i++) seed = mix64(seed, vm->s->rng[i]);
    rng_seed(S->rng, seed);
    if (S->guide) {
        search_expand(S, root, agent, obs, obs_len, n_act, legal);
        if (S->selfplay && S->noise > 0) {
            /* Dirichlet noise, so that self-play tries what the network
             * would not; never in play. */
            float *prior = S->prior + S->nodes[root].stat;
            int k = mask_count(legal, n_act);
            double alpha = k > 10 ? 10.0 / k : 1.0, d[MAX_ACT], sum = 0.0;
            for (int a = 0; a < n_act; a++)
                if ((legal >> a) & 1u) { d[a] = rng_gamma(S->rng, alpha); sum += d[a]; }
            for (int a = 0; a < n_act; a++)
                if ((legal >> a) & 1u)
                    prior[a] = (float)((1.0 - S->noise) * prior[a] +
                                       S->noise * (sum > 0 ? d[a] / sum : 1.0 / k));
        }
    }

    VM *sv = &S->vm;
    int ran = 0;
    g_searching = 1;
    for (int n = 0; n < S->playouts; n++, ran++) {
        if (S->slice > 0 && (n & 15) == 15 && now_sec() - t0 >= S->slice) break;
        memcpy(&S->st, &S->root, state_bytes);
        /* chance from here on is drawn afresh in every playout */
        rng_seed(S->st.rng, rng_next(S->rng));
        memcpy(sv->stack, S->root_stack, sizeof(double) * (size_t)sp);
        memcpy(sv->call_ip, S->root_calls, sizeof(int) * (size_t)csp);
        memcpy(sv->scratch, S->root_scratch, sizeof(sv->scratch));
        sv->sp = sp;
        sv->csp = csp;
        sv->insns = insns;
        sv->in_init = in_init;
        sv->halt = 0;
        sv->stopped = 0;
        sv->legal_pending = 0;
        sv->driver = &S->dr;
        memset(sv->agent_reward, 0, sizeof(sv->agent_reward));
        memset(S->paid, 0, sizeof(S->paid));
        S->npath = 0;
        S->rollout = 0;
        S->leafed = 0;

        int first = search_pick(S, root);
        search_step_in(S, root, first, 0.0);
        push(sv, (double)first);
        vm_exec(sv, ip);                /* the rest of the block the decision is in */
        if (!sv->stopped) {
            if (!in_init)
                for (int a = 0; a < p->nagents; a++) S->paid[a] += sv->agent_reward[a];
            while (!S->st.done && S->st.step < S->horizon) {
                vm_step(sv);
                if (sv->stopped) break;
                for (int a = 0; a < p->nagents; a++) S->paid[a] += sv->agent_reward[a];
            }
        }
        if (S->leafed && !sv->stopped)
            for (int a = 0; a < p->nagents; a++)
                S->leaf[a] = (1.0 - S->mix) * S->leaf[a] + S->mix * S->paid[a];
        const double *got = S->leafed ? S->leaf : S->paid;

        for (int k = 0; k < S->npath; k++) {
            SNode *nd = &S->nodes[S->path[k].node];
            double v = got[nd->agent] - S->path[k].before;
            if (v < S->low) S->low = v;
            if (v > S->high) S->high = v;
            nd->visits++;
            S->tries[nd->stat + S->path[k].action]++;
            S->total[nd->stat + S->path[k].action] += v;
        }
        S->played++;
    }
    g_searching = 0;
    S->searches++;

    /* the action tried most; between equals the one that did better, then the first */
    const SNode *nd = &S->nodes[root];
    int best = -1, all = 0;
    S->found_n = n_act;
    for (int a = 0; a < n_act; a++) {
        int t = S->tries[nd->stat + a];
        S->found_tries[a] = t;
        S->found_mean[a] = t ? S->total[nd->stat + a] / t : 0.0;
        if (!((legal >> a) & 1u)) continue;
        all += t;
        if (best < 0 || t > S->found_tries[best] ||
            (t == S->found_tries[best] && S->found_mean[a] > S->found_mean[best]))
            best = a;
    }
    if (S->selfplay) {
        if (S->rec) {
            GBuf *b = S->rec;
            if (b->n >= b->cap) {
                b->cap = b->cap ? b->cap * 2 : 1024;
                b->d = (GSample *)xrealloc(b->d, sizeof(GSample) * (size_t)b->cap);
            }
            GSample *g = &b->d[b->n++];
            memset(g, 0, sizeof(*g));
            g->game = S->game;
            g->ply = S->moves;
            g->agent = agent;
            g->n_act = n_act;
            g->legal = legal;
            g->obs_len = obs_len < MAX_OBS ? obs_len : MAX_OBS;
            memcpy(g->obs, obs, sizeof(double) * (size_t)g->obs_len);
            for (int a = 0; a < n_act; a++)
                g->pi[a] = all > 0 ? (float)S->found_tries[a] / (float)all : 0.0f;
            for (int a = 0; a < p->nagents; a++)
                g->z[a] = S->ep_paid[a] + (in_init ? 0.0 : vm->agent_reward[a]);
        }
        /* the opening of a self-play game is drawn, so that games differ */
        if (S->moves < S->sample_moves && all > 0) {
            int k = (int)(rng_double(S->rng) * all);
            for (int a = 0; a < n_act; a++) {
                if (!((legal >> a) & 1u)) continue;
                k -= S->found_tries[a];
                if (k < 0) { best = a; break; }
            }
        }
        S->moves++;
    }
    if (S->slice > 0 && ran > 0) S->cost += (now_sec() - t0) / ran;
    return best;
}

/* What the search just run found: how often each action was tried at the
 * decision, and the mean of what its agent went on to collect. */
static void search_found(const Search *S, int *tries, double *mean) {
    memcpy(tries, S->found_tries, sizeof(int) * (size_t)S->found_n);
    memcpy(mean, S->found_mean, sizeof(double) * (size_t)S->found_n);
}


static const char *const HIDDEN_NAME[] = { "nothing", "found", "unknown" };

/* A search with a network is version 2: the lines of version 1, then
 *   rollout SHARE
 *   guide NETS VSCALE GAMES DECISIONS ROUNDS SECONDS CORES PLAYOUTS
 *   of NET-OF-AGENT-0 NET-OF-AGENT-1 ...
 * and for each network
 *   net OBS ACTIONS VALUES HIDDEN SHARED
 *   in COUNT VALUE...      (one line per part of the observation; in 0 MEAN SCALE)
 *   w TOTAL, then that many weights. */
static void save_search(const char *path, const Program *p, const Search *S) {
    FILE *f = fopen(path, "w");
    if (!f) fatal(1, "hunch-vm: cannot write %s: %s", path, strerror(errno));
    const Guide *G = S->guide;
    fprintf(f, "HUNCHSEARCH %d\nworld %s\nplayouts %d\nexplore %.9g\nhidden %s\n"
            "checked %ld %ld %ld %ld\n", G ? 2 : 1, p->world, S->playouts, S->explore,
            HIDDEN_NAME[S->hidden], S->checked[0], S->checked[1], S->checked[2], S->checked[3]);
    if (G) {
        fprintf(f, "rollout %.9g\n", S->mix);
        fprintf(f, "guide %d %.17g %ld %ld %ld %.3f %d %d\nof", G->nnets, G->vscale, G->games,
                G->samples, G->rounds, G->sec, G->cores, G->playouts);
        for (int a = 0; a < p->nagents; a++) fprintf(f, " %d", G->net_of[a]);
        fputc('\n', f);
        for (int k = 0; k < G->nnets; k++) {
            const GNet *n = G->net[k];
            fprintf(f, "net %d %d %d %d %d\n", n->obs_dim, n->nact, n->nval, n->hid, n->shared);
            for (int d = 0; d < n->obs_dim; d++) {
                fprintf(f, "in %d", n->nv[d]);
                if (n->nv[d])
                    for (int j = 0; j < n->nv[d]; j++) fprintf(f, " %.17g", n->val[d][j]);
                else
                    fprintf(f, " %.17g %.17g", n->mean[d], n->scale[d]);
                fputc('\n', f);
            }
            fprintf(f, "w %d\n", n->total);
            for (int i = 0; i < n->total; i++)
                fprintf(f, "%.17g%c", n->w[i], i % 8 == 7 || i == n->total - 1 ? '\n' : ' ');
        }
    }
    if (fclose(f) != 0) fatal(1, "hunch-vm: cannot write %s: %s", path, strerror(errno));
}

/* The network of a version 2 file, after its `guide` tag; NULL if it does
 * not fit this world. */
static Guide *guide_read(FILE *f, const Program *p) {
    Guide *G = (Guide *)xcalloc(1, sizeof(Guide));
    char tag[16];
    if (fscanf(f, "%d %lf %ld %ld %ld %lf %d %d", &G->nnets, &G->vscale, &G->games, &G->samples,
               &G->rounds, &G->sec, &G->cores, &G->playouts) != 8 ||
        G->nnets < 1 || G->nnets > p->nagents || !(G->vscale > 0) || !isfinite(G->vscale))
        goto bad;
    if (fscanf(f, "%15s", tag) != 1 || strcmp(tag, "of") != 0) goto bad;
    for (int a = 0; a < p->nagents; a++)
        if (fscanf(f, "%d", &G->net_of[a]) != 1 || G->net_of[a] < 0 || G->net_of[a] >= G->nnets)
            goto bad;
    for (int k = 0; k < G->nnets; k++) {
        GNet *n = (GNet *)xcalloc(1, sizeof(GNet));
        G->net[k] = n;
        int total = 0;
        if (fscanf(f, "%15s %d %d %d %d %d", tag, &n->obs_dim, &n->nact, &n->nval, &n->hid,
                   &n->shared) != 6 || strcmp(tag, "net") != 0 || n->obs_dim < 0 ||
            n->obs_dim > MAX_OBS || n->nact < 1 || n->nact > MAX_ACT || n->nval != p->nagents ||
            n->hid < 1 || n->hid > GUIDE_MAX_HID || (n->shared != 0 && n->shared != 1))
            goto bad;
        for (int d = 0; d < n->obs_dim; d++) {
            if (fscanf(f, "%15s %d", tag, &n->nv[d]) != 2 || strcmp(tag, "in") != 0 ||
                n->nv[d] < 0 || n->nv[d] > GUIDE_ONEHOT)
                goto bad;
            for (int j = 0; j < n->nv[d]; j++)
                if (fscanf(f, "%lf", &n->val[d][j]) != 1) goto bad;
            if (!n->nv[d] && (fscanf(f, "%lf %lf", &n->mean[d], &n->scale[d]) != 2 ||
                              !(n->scale[d] > 0)))
                goto bad;
        }
        gnet_layout(n);
        if (fscanf(f, "%15s %d", tag, &total) != 2 || strcmp(tag, "w") != 0 || total != n->total)
            goto bad;
        n->w = (double *)xmalloc(sizeof(double) * (size_t)n->total);
        for (int i = 0; i < n->total; i++)
            if (fscanf(f, "%lf", &n->w[i]) != 1 || !isfinite(n->w[i])) goto bad;
    }
    for (int a = 0; a < p->nagents; a++) {
        const GNet *n = G->net[G->net_of[a]];
        if (n->obs_dim != p->agents[a].obs_dim || n->nact != p->agents[a].n_act) goto bad;
    }
    return G;
bad:
    guide_free(G);
    return NULL;
}

/* The rest of a HUNCHSEARCH file, after its first line. */
static Search *load_search(FILE *f, const Program *p, int version) {
    Guide *guide = NULL;
    double mix = 0.0;
    char tag[64], word[64];
    int playouts = 0, hidden = HIDDEN_UNKNOWN;
    double explore = 1.0;
    long checked[4] = {0, 0, 0, 0};
    if (fscanf(f, "%63s %63s", tag, word) != 2) return NULL;
    /* A search made for another world would run, and mean something else. */
    if (strcmp(tag, "world") != 0 || strcmp(word, p->world) != 0) return NULL;
    while (fscanf(f, "%63s", tag) == 1) {
        if (strcmp(tag, "playouts") == 0) {
            if (fscanf(f, "%d", &playouts) != 1) return NULL;
        } else if (strcmp(tag, "explore") == 0) {
            if (fscanf(f, "%lf", &explore) != 1) return NULL;
        } else if (strcmp(tag, "hidden") == 0) {
            if (fscanf(f, "%63s", word) != 1) return NULL;
            hidden = -1;
            for (int k = 0; k < 3; k++)
                if (strcmp(word, HIDDEN_NAME[k]) == 0) hidden = k;
            if (hidden < 0) return NULL;
        } else if (strcmp(tag, "checked") == 0) {
            if (fscanf(f, "%ld %ld %ld %ld", &checked[0], &checked[1], &checked[2],
                       &checked[3]) != 4) return NULL;
        } else if (strcmp(tag, "rollout") == 0 && version == 2) {
            if (fscanf(f, "%lf", &mix) != 1 || !(mix >= 0.0 && mix <= 1.0)) return NULL;
        } else if (strcmp(tag, "guide") == 0 && version == 2 && !guide) {
            guide = guide_read(f, p);
            if (!guide) return NULL;
        } else {
            guide_free(guide);
            return NULL;
        }
    }
    if (playouts < 1 || !(explore >= 0.0) || explore > 1e6 || (version == 2) != (guide != NULL)) {
        guide_free(guide);
        return NULL;
    }
    Search *S = search_new(p, playouts, explore, hidden);
    memcpy(S->checked, checked, sizeof(checked));
    S->guide = guide;
    S->mix = mix;
    return S;
}

/* ---- is anything hidden from the agents? ---- */

/* A search reads the whole state, so it is a fair policy only where each
 * agent is shown everything. The exact graph can prove that; a world too big
 * for the graph is tested by sampling instead. Random episodes are played,
 * and at every decision what the agent is shown (the agent, its observation
 * and which actions are legal) is noted against the instant it was shown in:
 * every slot of the world's memory, the step number, and the place in the
 * block (the same things a search tells positions apart by).
 *
 * - The same thing shown in two different instants, even once: something is
 *   hidden (HIDDEN_FOUND). The step number counts, so an agent that is not
 *   shown how far the episode has run is an agent with something hidden.
 * - Nothing is hidden (HIDDEN_NOTHING) only on evidence: at least
 *   SHOWN_MIN_REPEATS sampled positions showed something seen before, they
 *   were at least SHOWN_MIN_SHARE of all sampled positions, at least
 *   SHOWN_MIN_DISTINCT different things were shown more than once, and every
 *   one of those repeats was in the instant it had been in before.
 * - Otherwise nothing can be said (HIDDEN_UNKNOWN): observations that are
 *   measurements never repeat, and prove nothing. The share and the number
 *   of different things keep an opening that is the same in every episode
 *   (one thing shown, repeated thousands of times) from counting as evidence.
 *
 * It is evidence, not proof: random play repeats mostly the positions near
 * the start, so a world that hides something only where random play never
 * shows the same thing twice would pass. */
#define SHOWN_POSITIONS    40000L   /* decisions sampled */
#define SHOWN_EPISODES     20000L   /* or this many episodes */
#define SHOWN_SECONDS      3.0      /* or this long */
#define SHOWN_MIN_REPEATS  1000L
#define SHOWN_MIN_SHARE    0.05
#define SHOWN_MIN_DISTINCT 50L

typedef struct {
    Driver dr;          /* first, so a Driver * is a Shown * */
    long nslot;         /* a power of two */
    uint64_t *key, *i1, *i2;
    unsigned char *seen;        /* 0 empty, 1 shown once, 2 shown again */
    long positions, repeats, distinct, conflicts, episodes;
} Shown;

static int shown_decide(Driver *dr, VM *vm, int agent, const double *obs, int obs_len,
                        int n_act, uint32_t legal, const double *probs) {
    Shown *W = (Shown *)dr;
    (void)n_act; (void)probs;
    if (W->positions >= SHOWN_POSITIONS) return DRIVE_PASS;
    uint64_t h = tab_hash(agent, obs, obs_len, legal), h1, h2;
    search_key(vm, agent, &h1, &h2);
    W->positions++;
    long at = (long)(h & (uint64_t)(W->nslot - 1));
    while (W->seen[at] && W->key[at] != h) at = (at + 1) & (W->nslot - 1);
    if (!W->seen[at]) {
        W->seen[at] = 1;
        W->key[at] = h;
        W->i1[at] = h1;
        W->i2[at] = h2;
        return DRIVE_PASS;
    }
    W->repeats++;
    if (W->seen[at] == 1) { W->seen[at] = 2; W->distinct++; }
    if (W->i1[at] != h1 || W->i2[at] != h2) W->conflicts++;
    return DRIVE_PASS;
}

/* Sample the world with uniform random play. Returns HIDDEN_*; `out` gets
 * positions sampled, repeats, different things shown more than once, and
 * repeats in a different instant. */
static int shown_check(const Program *p, int horizon, uint64_t seed, long out[4]) {
    Shown W;
    memset(&W, 0, sizeof(W));
    W.dr.decide = shown_decide;
    W.nslot = 1;
    while (W.nslot < SHOWN_POSITIONS * 2) W.nslot *= 2;
    W.key = (uint64_t *)xmalloc(sizeof(uint64_t) * (size_t)W.nslot);
    W.i1 = (uint64_t *)xmalloc(sizeof(uint64_t) * (size_t)W.nslot);
    W.i2 = (uint64_t *)xmalloc(sizeof(uint64_t) * (size_t)W.nslot);
    W.seen = (unsigned char *)xcalloc((size_t)W.nslot, 1);
    State *s = (State *)xmalloc(sizeof(State));
    VM *vm = (VM *)xmalloc(sizeof(VM));
    vm_init(vm, p, s);
    vm->driver = &W.dr;
    double t0 = now_sec();
    while (W.positions < SHOWN_POSITIONS && W.episodes < SHOWN_EPISODES &&
           now_sec() - t0 < SHOWN_SECONDS) {
        run_episode(vm, NULL, NULL, mix64(seed + 0x5407ULL, (uint64_t)W.episodes), 0,
                    (int)W.episodes, NULL, horizon);
        W.episodes++;
    }
    out[0] = W.positions; out[1] = W.repeats; out[2] = W.distinct; out[3] = W.conflicts;
    free(W.key); free(W.i1); free(W.i2); free(W.seen); free(vm); free(s);
    if (W.conflicts > 0) return HIDDEN_FOUND;
    if (W.repeats >= SHOWN_MIN_REPEATS && W.distinct >= SHOWN_MIN_DISTINCT &&
        (double)W.repeats >= SHOWN_MIN_SHARE * (double)W.positions)
        return HIDDEN_NOTHING;
    return HIDDEN_UNKNOWN;
}

/* How many playouts a second a search of this world gets through on this
 * machine: episodes are played by a search that is given `slice` seconds at
 * each decision, until `searches` decisions have been searched or `budget`
 * seconds have gone. 0 if no decision was reached.
 *
 * A playout from late in an episode is short and one from its start is long,
 * so the figure is taken per decision: the time a playout took at each
 * decision searched, averaged over the decisions, and turned over. A search
 * that makes that many playouts at every decision then takes a second on
 * average, more near the start of an episode and less near its end. It
 * covers the decisions of the first episode or two; an episode much longer
 * than `searches` decisions is measured at its start only. */
static double search_measure(const Program *p, int horizon, double explore, uint64_t seed,
                             double slice, int searches, double budget, Guide *guide,
                             double mix) {
    Search *M = search_new(p, 100000, explore, HIDDEN_UNKNOWN);
    M->horizon = horizon;
    M->guide = guide;
    M->mix = guide ? mix : 0.0;
    M->slice = slice;
    State *s = (State *)xmalloc(sizeof(State));
    VM *vm = (VM *)xmalloc(sizeof(VM));
    vm_init(vm, p, s);
    vm->search = M;
    double t0 = now_sec();
    for (uint64_t e = 0; M->searches < searches && now_sec() - t0 < budget; e++) {
        state_reset(p, s, seed + e * 7919ULL);
        s->ep = 0;
        vm_run(vm, p->init_entry);
        while (!s->done && s->step < horizon && M->searches < searches &&
               now_sec() - t0 < budget)
            vm_step(vm);
    }
    double rate = M->cost > 0 ? (double)M->searches / M->cost : 0.0;
    free(vm); free(s);
    M->guide = NULL;            /* the caller's */
    search_free(M);
    return rate;
}

/* ------------------------------------------------------------------ */
/* Learning to search: self-play                                       */
/* ------------------------------------------------------------------ */

/* train makes a search's network by self-play. A round is a number of games
 * played by the search against itself with the network it has so far (with
 * noise at the root and the first moves drawn, so that games differ); every
 * decision of them is kept: what the agent was shown, where the search spent
 * its playouts, and what each agent went on to collect. The network is then
 * trained on the most recent decisions to say where a search would spend its
 * playouts and what each agent will collect, and the next round is played
 * with it. A better network makes a better search, whose games teach a better
 * network.
 *
 * Games are spread over processes, one per core, and gathered in the order
 * they were numbered, so what train makes depends on the seed and the number
 * of games, not on how many cores played them. */

/* What random play shows each agent: which parts of it take few values. */
typedef struct {
    Driver dr;
    Guide *G;
    int many[MAX_AGENTS][MAX_OBS];
    double sum[MAX_AGENTS][MAX_OBS], sq[MAX_AGENTS][MAX_OBS];
    long count[MAX_AGENTS], positions;
} GSeen;

static int gseen_decide(Driver *dr, VM *vm, int agent, const double *obs, int obs_len,
                        int n_act, uint32_t legal, const double *probs) {
    GSeen *W = (GSeen *)dr;
    (void)vm; (void)n_act; (void)legal; (void)probs;
    int k = W->G->net_of[agent];
    GNet *n = W->G->net[k];
    for (int d = 0; d < n->obs_dim; d++) {
        double v = d < obs_len ? obs[d] : 0.0;
        W->sum[k][d] += v;
        W->sq[k][d] += v * v;
        if (W->many[k][d]) continue;
        int j = 0;
        while (j < n->nv[d] && n->val[d][j] != v) j++;
        if (j < n->nv[d]) continue;
        if (n->nv[d] < GUIDE_ONEHOT) n->val[d][n->nv[d]++] = v;
        else W->many[k][d] = 1;
    }
    W->count[k]++;
    W->positions++;
    return DRIVE_PASS;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* A new network for this world, its inputs settled by random play. */
static Guide *guide_new(const Program *p, int horizon, uint64_t seed, int hid) {
    Guide *G = (Guide *)xcalloc(1, sizeof(Guide));
    int same = 1;
    for (int a = 1; a < p->nagents; a++)
        if (p->agents[a].obs_dim != p->agents[0].obs_dim || p->agents[a].n_act != p->agents[0].n_act)
            same = 0;
    G->nnets = same ? 1 : p->nagents;
    for (int a = 0; a < p->nagents; a++) G->net_of[a] = same ? 0 : a;
    for (int k = 0; k < G->nnets; k++) {
        GNet *n = (GNet *)xcalloc(1, sizeof(GNet));
        n->obs_dim = p->agents[k].obs_dim;
        n->nact = p->agents[k].n_act;
        n->nval = p->nagents;
        n->hid = hid < 1 ? 1 : hid > GUIDE_MAX_HID ? GUIDE_MAX_HID : hid;
        n->shared = same && p->nagents > 1;
        G->net[k] = n;
    }
    GSeen *W = (GSeen *)xcalloc(1, sizeof(GSeen));
    W->dr.decide = gseen_decide;
    W->G = G;
    State *s = (State *)xmalloc(sizeof(State));
    VM *vm = (VM *)xmalloc(sizeof(VM));
    vm_init(vm, p, s);
    vm->driver = &W->dr;
    double most = 0.0, ret[MAX_AGENTS], t0 = now_sec();
    for (long e = 0; e < 4000 && W->positions < 40000 && now_sec() - t0 < 2.0; e++) {
        run_episode(vm, NULL, NULL, mix64(seed + 0x6E7ULL, (uint64_t)e), 0, (int)e, ret, horizon);
        for (int a = 0; a < p->nagents; a++)
            if (fabs(ret[a]) > most) most = fabs(ret[a]);
    }
    G->vscale = most > 1e-9 ? most : 1.0;
    uint64_t rng[4];
    rng_seed(rng, seed ^ 0x6E7C0DEULL);
    for (int k = 0; k < G->nnets; k++) {
        GNet *n = G->net[k];
        for (int d = 0; d < n->obs_dim; d++) {
            if (W->many[k][d] || W->count[k] == 0) {
                double c = W->count[k] > 0 ? (double)W->count[k] : 1.0;
                double m = W->sum[k][d] / c, var = W->sq[k][d] / c - m * m;
                n->nv[d] = 0;
                n->mean[d] = m;
                n->scale[d] = var > 1e-18 ? sqrt(var) : 1.0;
            } else {
                qsort(n->val[d], (size_t)n->nv[d], sizeof(double), cmp_double);
            }
        }
        gnet_layout(n);
        n->w = (double *)xcalloc((size_t)n->total, sizeof(double));
        n->g = (double *)xcalloc((size_t)n->total, sizeof(double));
        n->m = (double *)xcalloc((size_t)n->total, sizeof(double));
        n->v = (double *)xcalloc((size_t)n->total, sizeof(double));
        int H = n->hid;
        double r1 = sqrt(6.0 / (n->obs_dim + n->shared + 1.0)), r2 = sqrt(6.0 / H);
        for (int i = 0; i < n->nin * H; i++) n->w[n->o_w1 + i] = (rng_double(rng) * 2 - 1) * r1;
        for (int i = 0; i < H * H; i++) n->w[n->o_w2 + i] = (rng_double(rng) * 2 - 1) * r2;
        for (int i = 0; i < n->nact * H; i++) n->w[n->o_wp + i] = (rng_double(rng) * 2 - 1) * 0.01;
        for (int i = 0; i < n->nval * H; i++) n->w[n->o_wv + i] = (rng_double(rng) * 2 - 1) * 0.01;
    }
    free(vm); free(s); free(W);
    return G;
}

/* Gradient of one decision: cross-entropy to where the search spent its
 * playouts, and squared error to what each agent went on to collect. */
static void gnet_backward(GNet *n, const GPass *ps, int n_act, uint32_t legal, const float *pi,
                          const double *z) {
    const int H = n->hid;
    const double *w = n->w;
    double *g = n->g;
    double d2[GUIDE_MAX_HID], d1[GUIDE_MAX_HID];
    memset(d2, 0, sizeof(double) * (size_t)H);
    for (int a = 0; a < n_act; a++) {
        if (!((legal >> a) & 1u)) continue;
        double d = ps->probs[a] - pi[a];
        g[n->o_bp + a] += d;
        double *gr = g + n->o_wp + (size_t)a * H;
        const double *wr = w + n->o_wp + (size_t)a * H;
        for (int i = 0; i < H; i++) { gr[i] += d * ps->h2[i]; d2[i] += d * wr[i]; }
    }
    for (int j = 0; j < n->nval; j++) {
        double d = ps->value[j] - z[j];
        g[n->o_bv + j] += d;
        double *gr = g + n->o_wv + (size_t)j * H;
        const double *wr = w + n->o_wv + (size_t)j * H;
        for (int i = 0; i < H; i++) { gr[i] += d * ps->h2[i]; d2[i] += d * wr[i]; }
    }
    memset(d1, 0, sizeof(double) * (size_t)H);
    for (int i = 0; i < H; i++) {
        if (ps->h2[i] <= 0) continue;
        double d = d2[i];
        g[n->o_b2 + i] += d;
        double *gr = g + n->o_w2 + (size_t)i * H;
        const double *wr = w + n->o_w2 + (size_t)i * H;
        for (int j = 0; j < H; j++) { gr[j] += d * ps->h1[j]; d1[j] += d * wr[j]; }
    }
    for (int i = 0; i < H; i++) {
        if (ps->h1[i] <= 0) d1[i] = 0.0;
        g[n->o_b1 + i] += d1[i];
    }
    for (int k = 0; k < ps->n; k++) {
        double *gr = g + n->o_w1 + (size_t)ps->idx[k] * H;
        double xv = ps->x[k];
        for (int i = 0; i < H; i++) gr[i] += xv * d1[i];
    }
}

static void gnet_adam(GNet *n, double lr, double scale, double decay) {
    n->opt_step++;
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    double c1 = 1.0 - pow(b1, (double)n->opt_step), c2 = 1.0 - pow(b2, (double)n->opt_step);
    for (int i = 0; i < n->total; i++) {
        double g = n->g[i] * scale;
        n->m[i] = b1 * n->m[i] + (1 - b1) * g;
        n->v[i] = b2 * n->v[i] + (1 - b2) * g * g;
        n->w[i] -= lr * ((n->m[i] / c1) / (sqrt(n->v[i] / c2) + eps) + decay * n->w[i]);
        n->g[i] = 0.0;
    }
}

/* `steps` minibatches drawn from the decisions kept; the mean losses of the
 * last of them go to loss[0] (policy) and loss[1] (value). */
static void guide_learn(Guide *G, const GBuf *buf, long steps, int batch, double lr,
                        uint64_t rng[4], double *loss) {
    GPass ps;
    long seen[MAX_AGENTS];
    double lp = 0.0, lv = 0.0;
    long nl = 0;
    for (long st = 0; st < steps && buf->n > 0; st++) {
        memset(seen, 0, sizeof(seen));
        for (int b = 0; b < batch; b++) {
            const GSample *x = &buf->d[(long)(rng_double(rng) * buf->n)];
            int k = G->net_of[x->agent];
            GNet *n = G->net[k];
            double z[MAX_AGENTS];
            for (int a = 0; a < n->nval; a++) z[a] = x->z[a] / G->vscale;
            gnet_encode(n, x->agent, x->obs, x->obs_len, &ps);
            gnet_forward(n, &ps, x->n_act, x->legal);
            gnet_backward(n, &ps, x->n_act, x->legal, x->pi, z);
            seen[k]++;
            if (st >= steps - 8) {
                for (int a = 0; a < x->n_act; a++)
                    if (x->pi[a] > 0) lp -= x->pi[a] * log(ps.probs[a] > 1e-12 ? ps.probs[a] : 1e-12);
                for (int a = 0; a < n->nval; a++) lv += (ps.value[a] - z[a]) * (ps.value[a] - z[a]);
                nl++;
            }
        }
        for (int k = 0; k < G->nnets; k++)
            if (seen[k]) gnet_adam(G->net[k], lr, 1.0 / (double)seen[k], 1e-4);
    }
    loss[0] = nl ? lp / nl : 0.0;
    loss[1] = nl ? lv / nl : 0.0;
}

/* One game of self-play; its decisions are added to `buf`. */
static void selfplay_game(const Program *p, VM *vm, Search *S, int horizon, uint64_t seed,
                          long game, GBuf *buf) {
    State *s = vm->s;
    long first = buf->n;
    state_reset(p, s, mix64(seed ^ 0x5E1F9A7EULL, (uint64_t)game));
    s->ep = game;
    S->game = game;
    S->moves = 0;
    S->rec = buf;
    memset(S->ep_paid, 0, sizeof(S->ep_paid));
    vm_run(vm, p->init_entry);
    while (!s->done && s->step < horizon) {
        vm_step(vm);
        for (int a = 0; a < p->nagents; a++) S->ep_paid[a] += vm->agent_reward[a];
    }
    for (long k = first; k < buf->n; k++)
        for (int a = 0; a < p->nagents; a++) buf->d[k].z[a] = S->ep_paid[a] - buf->d[k].z[a];
}

static int cmp_sample(const void *a, const void *b) {
    const GSample *x = (const GSample *)a, *y = (const GSample *)b;
    if (x->game != y->game) return x->game < y->game ? -1 : 1;
    return (x->ply > y->ply) - (x->ply < y->ply);
}

static void selfplay_games(const Program *p, Search *S, int horizon, uint64_t seed, long first,
                           long games, long stride, GBuf *buf) {
    State *s = (State *)xmalloc(sizeof(State));
    VM *vm = (VM *)xmalloc(sizeof(VM));
    vm_init(vm, p, s);
    vm->search = S;
    for (long g = first; g < first + games; g += stride)
        selfplay_game(p, vm, S, horizon, seed, g, buf);
    free(vm); free(s);
}

#define MAX_CORES 256
#define GUIDE_TRAIN_SEC 600.0   /* how long train plays itself unless told */
#define GUIDE_EXPLORE   1.5     /* PUCT's constant, with values scaled to the range seen */
#define GUIDE_ROLLOUT   0.5     /* a leaf's value: this share one random playout, the rest the network */

static int cores_online(void) {
#ifdef _SC_NPROCESSORS_ONLN
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : n > MAX_CORES ? MAX_CORES : (int)n;
#else
    return 1;
#endif
}

/* Games first .. first + games - 1, on up to `cores` processes. */
static void selfplay_round(const Program *p, Search *S, int horizon, uint64_t seed, long first,
                           long games, int cores, GBuf *buf) {
    long start = buf->n;
    if (cores > games) cores = (int)games;
    if (cores > MAX_CORES) cores = MAX_CORES;
    if (cores <= 1) {
        selfplay_games(p, S, horizon, seed, first, games, 1, buf);
        return;
    }
    FILE *tf[MAX_CORES];
    pid_t pid[MAX_CORES];
    fflush(NULL);
    for (int w = 0; w < cores; w++) {
        tf[w] = tmpfile();
        if (!tf[w]) fatal(1, "hunch-vm: no temporary file for self-play: %s", strerror(errno));
        pid[w] = fork();
        if (pid[w] < 0) fatal(1, "hunch-vm: cannot start a self-play process: %s", strerror(errno));
        if (pid[w] == 0) {
            GBuf mine = {NULL, 0, 0};
            selfplay_games(p, S, horizon, seed, first + w, games - w, cores, &mine);
            int ok = fwrite(&mine.n, sizeof(long), 1, tf[w]) == 1 &&
                     (mine.n == 0 || fwrite(mine.d, sizeof(GSample), (size_t)mine.n, tf[w]) ==
                                         (size_t)mine.n) &&
                     fflush(tf[w]) == 0;
            _exit(ok ? 0 : 1);
        }
    }
    int failed = 0;
    for (int w = 0; w < cores; w++) {
        int status = 0;
        if (waitpid(pid[w], &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            failed = 1;
        long n = 0;
        rewind(tf[w]);
        if (!failed && fread(&n, sizeof(long), 1, tf[w]) == 1 && n >= 0) {
            if (buf->n + n > buf->cap) {
                while (buf->n + n > buf->cap) buf->cap = buf->cap ? buf->cap * 2 : 1024;
                buf->d = (GSample *)xrealloc(buf->d, sizeof(GSample) * (size_t)buf->cap);
            }
            if (fread(buf->d + buf->n, sizeof(GSample), (size_t)n, tf[w]) != (size_t)n) failed = 1;
            else buf->n += n;
        } else {
            failed = 1;
        }
        fclose(tf[w]);
    }
    if (failed) fatal(1, "hunch-vm: a self-play process failed");
    qsort(buf->d + start, (size_t)(buf->n - start), sizeof(GSample), cmp_sample);
}

typedef struct {
    long games;         /* stop after this many (rounded up to whole rounds); <= 0: no limit */
    double seconds;     /* or after this long, checked between rounds */
    int cores, round, playouts, hid, sample_moves, batch;
    long window;        /* decisions kept to learn from: the most recent */
    double reuse;       /* times each decision is drawn into a minibatch, on average */
    double lr, noise, explore;
} Learn;

/* Self-play until the games or the time run out; the network it made. */
static Guide *guide_train(const Program *p, int horizon, uint64_t seed, int hidden,
                          const Learn *L) {
    double t0 = now_sec();
    Guide *G = guide_new(p, horizon, seed, L->hid);
    Search *T = search_new(p, L->playouts, L->explore, hidden);
    T->horizon = horizon;
    T->guide = G;
    T->selfplay = 1;
    T->noise = L->noise;
    T->sample_moves = L->sample_moves;
    GBuf buf = {NULL, 0, 0};
    uint64_t rng[4];
    rng_seed(rng, seed ^ 0x7EA4C4E5ULL);
    long games = 0, samples = 0;
    int round = L->round < 1 ? 1 : L->round;
    while ((L->games <= 0 || games < L->games) && (games == 0 || now_sec() - t0 < L->seconds)) {
        long before = buf.n;
        selfplay_round(p, T, horizon, seed, games, round, L->cores, &buf);
        long fresh = buf.n - before;
        /* what the first agent collected, and how many games nobody won */
        double first = 0.0;
        long played = 0, even = 0;
        for (long k = before; k < buf.n; k++) {
            const GSample *x = &buf.d[k];
            if (x->ply != 0) continue;
            int zero = 1;
            for (int a = 0; a < p->nagents; a++) zero &= x->z[a] == 0.0;
            first += x->z[0];
            even += zero;
            played++;
        }
        games += round;
        samples += fresh;
        G->rounds++;
        if (buf.n > L->window) {
            long drop = buf.n - L->window;
            memmove(buf.d, buf.d + drop, sizeof(GSample) * (size_t)(buf.n - drop));
            buf.n -= drop;
        }
        long steps = (long)ceil((double)fresh * L->reuse / L->batch);
        double loss[2];
        guide_learn(G, &buf, steps, L->batch, L->lr, rng, loss);
        if (g_progress)
            fprintf(stderr, "games %ld %.1f %.6f %.6f %.6f %.6f\n", games, now_sec() - t0,
                    played ? first / played : 0.0, played ? (double)even / played : 0.0,
                    loss[0], loss[1]);
    }
    T->guide = NULL;
    search_free(T);
    free(buf.d);
    /* only play needs the weights */
    for (int k = 0; k < G->nnets; k++) {
        GNet *n = G->net[k];
        free(n->g); free(n->m); free(n->v);
        n->g = n->m = n->v = NULL;
    }
    G->games = games;
    G->samples = samples;
    G->sec = now_sec() - t0;
    G->cores = L->cores;
    G->playouts = L->playouts;
    return G;
}

/* ------------------------------------------------------------------ */
/* Policies: a network per agent, one table, or a search               */
/* ------------------------------------------------------------------ */

typedef struct {
    Net *nets[MAX_AGENTS];
    Table *tab;
    Search *search;
} Policy;

static void save_table(const char *path, const Program *p, const Table *T) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "hunch-vm: cannot write %s: %s\n", path, strerror(errno));
        exit(1);
    }
    fprintf(f, "HUNCHTAB 1\nworld %s\nrows %ld\n", p->world, T->n);
    double probs[MAX_ACT];
    for (long id = 0; id < T->n; id++) {
        fprintf(f, "%d %d %u %d", T->agent[id], T->nact[id], T->mask[id], T->klen[id]);
        for (int i = 0; i < T->klen[id]; i++) fprintf(f, " %.17g", T->keys[T->koff[id] + i]);
        tab_average(T, id, probs);
        for (int a = 0; a < T->nact[id]; a++) fprintf(f, " %.9g", probs[a]);
        fputc('\n', f);
    }
    fclose(f);
}

static Table *load_table(FILE *f, const Program *p) {
    char tag[64], world[64];
    long rows = 0;
    if (fscanf(f, "%63s %63s", tag, world) != 2) return NULL;
    /* A table from another world can have the same shapes and mean nothing. */
    if (strcmp(world, p->world) != 0) return NULL;
    if (fscanf(f, "%63s %ld", tag, &rows) != 2 || rows < 0) return NULL;
    Table *T = tab_new();
    for (long r = 0; r < rows; r++) {
        int agent, nact, klen;
        unsigned mask;
        double obs[MAX_OBS];
        if (fscanf(f, "%d %d %u %d", &agent, &nact, &mask, &klen) != 4) { tab_free(T); return NULL; }
        if (agent < 0 || agent >= p->nagents || nact < 1 || nact > MAX_ACT || klen < 0 ||
            klen > MAX_OBS || nact != p->agents[agent].n_act || klen != p->agents[agent].obs_dim) {
            tab_free(T);
            return NULL;
        }
        for (int i = 0; i < klen; i++)
            if (fscanf(f, "%lf", &obs[i]) != 1) { tab_free(T); return NULL; }
        long id = tab_find(T, agent, obs, klen, nact, (uint32_t)mask, 1);
        for (int a = 0; a < nact; a++)
            if (fscanf(f, "%lf", &T->ssum[T->aoff[id] + a]) != 1) { tab_free(T); return NULL; }
    }
    return T;
}

/* Read a policy file of any kind. 1 on success. */
static int policy_load(const char *path, const Program *p, Policy *pol) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char tag[64];
    int version = 0;
    if (fscanf(f, "%63s %d", tag, &version) != 2) { fclose(f); return 0; }
    if (strcmp(tag, "HUNCHTAB") == 0) {
        pol->tab = load_table(f, p);
        fclose(f);
        return pol->tab != NULL;
    }
    if (strcmp(tag, "HUNCHSEARCH") == 0) {
        pol->search = version == 1 || version == 2 ? load_search(f, p, version) : NULL;
        fclose(f);
        return pol->search != NULL;
    }
    fclose(f);
    for (int i = 0; i < p->nagents; i++)
        if (!pol->nets[i])
            pol->nets[i] = net_new(p->agents[i].obs_dim, p->agents[i].n_act, 1);
    return load_policy(path, p, pol->nets);
}

static void policy_attach(VM *vm, const Policy *pol) {
    for (int i = 0; i < vm->p->nagents; i++) vm->nets[i] = pol ? pol->nets[i] : NULL;
    vm->tab = pol ? pol->tab : NULL;
    vm->tab_mode = vm->tab ? TAB_AVERAGE : TAB_OFF;
    vm->tab_create = 0;
    vm->search = pol ? pol->search : NULL;
}

static void policy_free(Policy *pol) {
    for (int i = 0; i < MAX_AGENTS; i++) net_free(pol->nets[i]);
    tab_free(pol->tab);
    search_free(pol->search);
}

/* ------------------------------------------------------------------ */
/* Plans: run one block with its first events decided in advance       */
/* ------------------------------------------------------------------ */

/* An event is a decision, or a randint when the plan counts chance. A plan
 * names the outcome of the first `nplan` events of a block. What happens at
 * the next one is `stop`: let it through, or end the block there and report
 * what the event was, so the caller can try each outcome in turn. The state
 * holds the random stream, so replaying a block from the same snapshot under
 * the same plan repeats it exactly. That is all a fork is. */
#define MAX_TRAIL 256

enum { STOP_NEVER = 0, STOP_ALL, STOP_AGENT };

typedef struct {
    Driver dr;                  /* first, so a Driver * is a Plan * */
    const int *plan;
    int nplan;
    int seen;                   /* events met so far */
    int stop, stop_agent;
    int chance_too;
    /* the event it stopped at */
    int ev_chance, ev_agent, ev_n, ev_lo, ev_obs_len;
    uint32_t ev_legal;
    long ev_row;
    double ev_probs[MAX_ACT];
    double ev_obs[MAX_OBS];
    /* the action every decision of the block ended with, in order */
    int trail[MAX_TRAIL];
    int ntrail;
    /* regret training: add the strategy of each decision past the plan to its
     * running average, with this weight (0 = leave the averages alone) */
    double avg_weight;
    /* why: start a new random stream right after decision number reseed_at */
    int reseed_at;
    uint64_t reseed;
} Plan;

static int plan_decide(Driver *dr, VM *vm, int agent, const double *obs, int obs_len,
                       int n_act, uint32_t legal, const double *probs) {
    Plan *pl = (Plan *)dr;
    int k = pl->seen;
    if (k < pl->nplan) { pl->seen++; return pl->plan[k]; }
    if (pl->stop == STOP_ALL || (pl->stop == STOP_AGENT && agent == pl->stop_agent)) {
        pl->ev_chance = 0;
        pl->ev_agent = agent;
        pl->ev_n = n_act;
        pl->ev_legal = legal;
        pl->ev_obs_len = obs_len;
        pl->ev_row = vm->tab_row;
        memcpy(pl->ev_probs, probs, sizeof(double) * (size_t)n_act);
        memcpy(pl->ev_obs, obs, sizeof(double) * (size_t)obs_len);
        return DRIVE_STOP;
    }
    pl->seen++;
    return DRIVE_PASS;
}

static void plan_decided(Driver *dr, VM *vm, int agent, int n_act, uint32_t legal,
                         const double *probs, int action) {
    Plan *pl = (Plan *)dr;
    (void)agent;
    int k = pl->seen - 1;
    if (!pl->chance_too) {
        if (pl->ntrail >= MAX_TRAIL) die("more than 256 decisions in one step");
        pl->trail[pl->ntrail++] = action;
    }
    if (pl->avg_weight > 0.0 && k >= pl->nplan && vm->tab_row >= 0 &&
        vm->tab_mode == TAB_CURRENT) {
        double *sum = vm->tab->ssum + vm->tab->aoff[vm->tab_row];
        for (int a = 0; a < n_act; a++)
            if ((legal >> a) & 1u) sum[a] += pl->avg_weight * probs[a];
    }
    if (pl->reseed && k == pl->reseed_at) rng_seed(vm->s->rng, pl->reseed);
}

static int plan_chance(Driver *dr, VM *vm, int lo, int count) {
    Plan *pl = (Plan *)dr;
    (void)vm;
    int k = pl->seen;
    if (k < pl->nplan) { pl->seen++; return pl->plan[k]; }
    if (pl->stop == STOP_ALL) {
        pl->ev_chance = 1;
        pl->ev_n = count;
        pl->ev_lo = lo;
        return DRIVE_STOP;
    }
    pl->seen++;
    return DRIVE_PASS;
}

static void plan_init(Plan *pl, const int *plan, int nplan, int stop, int stop_agent,
                      int chance_too) {
    memset(pl, 0, sizeof(*pl));
    pl->dr.decide = plan_decide;
    pl->dr.decided = plan_decided;
    pl->dr.chance = chance_too ? plan_chance : NULL;
    pl->dr.no_rand = chance_too;
    pl->plan = plan;
    pl->nplan = nplan;
    pl->stop = stop;
    pl->stop_agent = stop_agent;
    pl->chance_too = chance_too;
    pl->reseed_at = -1;
}

/* Run init or one step from `snap` under the plan. 1: the block ran to its
 * end. 0: it stopped at an event. -1: it stopped at a rand(). */
static int plan_run(VM *vm, Plan *pl, const State *snap, int is_init) {
    *vm->s = *snap;
    vm->driver = &pl->dr;
    if (is_init) {
        vm_run(vm, vm->p->init_entry);
        /* The runtime drops what init pays, here as everywhere. */
        memset(vm->agent_reward, 0, sizeof(vm->agent_reward));
    } else {
        vm_step(vm);
    }
    vm->driver = NULL;
    int stopped = vm->stopped;
    vm->stopped = 0;
    return stopped == 0 ? 1 : stopped == 2 ? -1 : 0;
}

/* Play on from where the VM stands to the end of the episode, with no plan.
 * Returns what `agent` collects on the way. */
static double play_out(VM *vm, int agent, int horizon) {
    double total = 0.0;
    while (!vm->s->done && vm->s->step < horizon) {
        vm_step(vm);
        total += vm->agent_reward[agent];
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Every outcome: the graph of all a world can do                      */
/* ------------------------------------------------------------------ */

/* When a world is small enough, its policy is not sampled but weighed. Every
 * chance outcome and every action is followed once, by forking at each event,
 * and every number about the policy is then a sum over what was found.
 *
 * Two runs that start a step with the same declared state and step number are
 * the same from there on: a local is always written before it is read, so
 * nothing else is carried from one step to the next. Such runs share one
 * node, which is what keeps a corridor walked for thirty steps at a few
 * hundred nodes instead of two to the thirtieth. `--no-merge` turns that off,
 * as a check: both ways must give the same numbers. */

enum { T_END = 0, T_CHANCE, T_DECIDE };

typedef struct {
    int kind, agent, nkid;
    long kid;           /* its first edge */
    long iset;          /* a decision: what the agent is shown, as a row of `isets` */
} TNode;

enum { TREE_OK = 0, TREE_TOO_BIG, TREE_RAND };

typedef struct {
    const Program *p;
    VM *vm;
    int horizon, nag, nstate, merge;
    TNode *n;
    long nn, cap;
    /* edges: the child, the action or outcome that leads to it, and what is
     * paid on the way (an index into `rews`, or -1 for nothing) */
    long *kids, *krew;
    int *kact;
    long nkids, capkids;
    double *rews;
    long nrews, caprews;
    long root, root_rew;
    long *order;        /* every node, each after all its children */
    long norder;
    /* Interns (agent, observation, legal moves). `ssum` holds the policy's
     * probabilities there; `regret` is unused. */
    Table *isets;
    /* Step-start states already met. A row's two values are the node the
     * state leads to and the pay on the way there (as an index into `rews`). */
    Table *memo;
    double *keybuf;
    long max_nodes;
    int fail, rand_line, rand_col;
} Tree;

static long tree_node(Tree *T, int kind) {
    if (T->nn >= T->max_nodes) { T->fail = TREE_TOO_BIG; return -1; }
    if (T->nn >= T->cap) {
        T->cap = T->cap ? T->cap * 2 : 1024;
        T->n = (TNode *)xrealloc(T->n, sizeof(TNode) * (size_t)T->cap);
        T->order = (long *)xrealloc(T->order, sizeof(long) * (size_t)T->cap);
    }
    long id = T->nn++;
    TNode *n = &T->n[id];
    memset(n, 0, sizeof(*n));
    n->kind = kind;
    n->iset = -1;
    return id;
}

/* Keep one payment per agent; -1 when nobody is paid anything. */
static long tree_pay(Tree *T, const double *pay) {
    int any = 0;
    for (int a = 0; a < T->nag; a++)
        if (pay[a] != 0.0) any = 1;
    if (!any) return -1;
    if (T->nrews + T->nag > T->caprews) {
        T->caprews = T->caprews ? T->caprews * 2 : 1024;
        T->rews = (double *)xrealloc(T->rews, sizeof(double) * (size_t)T->caprews);
    }
    long at = T->nrews;
    memcpy(T->rews + at, pay, sizeof(double) * (size_t)T->nag);
    T->nrews += T->nag;
    return at;
}

/* A fork can be this many events deep before the walk gives up. */
#define TREE_MAX_DEPTH 20000

/* The node reached from `from` by a block run under `plan`, and in *pay_out
 * what is paid on the way. Blocks that end without an event are run through. */
static long tree_arrive(Tree *T, const State *from, const int *plan, int nplan, int in_init,
                        long *pay_out, int depth) {
    *pay_out = -1;
    if (T->fail) return -1;
    if (depth > TREE_MAX_DEPTH) { T->fail = TREE_TOO_BIG; return -1; }
    VM *vm = T->vm;
    State *snap = (State *)xmalloc(sizeof(State));
    *snap = *from;
    double pend[MAX_AGENTS] = {0};
    long node = -1;
    /* the step-start states passed on the way, with what had been paid by then */
    long *brow = NULL;
    double *bpend = NULL;
    int nb = 0, capb = 0;
    for (;;) {
        Plan *pl = (Plan *)xmalloc(sizeof(Plan));
        plan_init(pl, plan, nplan, STOP_ALL, -1, 1);
        int r = plan_run(vm, pl, snap, in_init);
        if (r < 0) {
            T->fail = TREE_RAND;
            T->rand_line = vm->rand_line;
            T->rand_col = vm->rand_col;
            free(pl);
            break;
        }
        if (r == 1) {
            free(pl);
            if (!in_init)
                for (int a = 0; a < T->nag; a++) pend[a] += vm->agent_reward[a];
            *snap = *vm->s;
            in_init = 0;
            plan = NULL;
            nplan = 0;
            if (snap->done || snap->step >= T->horizon) {
                node = tree_node(T, T_END);
                if (node >= 0) T->order[T->norder++] = node;
                break;
            }
            if (T->merge) {
                int klen = T->nstate + 1;
                memcpy(T->keybuf, snap->v, sizeof(double) * (size_t)T->nstate);
                T->keybuf[T->nstate] = (double)snap->step;
                long before = T->memo->n;
                long row = tab_find(T->memo, 0, T->keybuf, klen, 2, 0, 1);
                if (T->memo->n == before) {
                    const double *m = T->memo->regret + T->memo->aoff[row];
                    node = (long)m[0];
                    long extra = (long)m[1];
                    if (extra >= 0)
                        for (int a = 0; a < T->nag; a++) pend[a] += T->rews[extra + a];
                    break;
                }
                if (nb >= capb) {
                    capb = capb ? capb * 2 : 4;
                    brow = (long *)xrealloc(brow, sizeof(long) * (size_t)capb);
                    bpend = (double *)xrealloc(bpend, sizeof(double) * (size_t)capb * MAX_AGENTS);
                }
                brow[nb] = row;
                memcpy(bpend + (size_t)nb * MAX_AGENTS, pend, sizeof(pend));
                nb++;
            }
            continue;
        }
        int ev_chance = pl->ev_chance, ev_n = pl->ev_n;
        uint32_t ev_legal = pl->ev_legal;
        node = tree_node(T, ev_chance ? T_CHANCE : T_DECIDE);
        if (node < 0) { free(pl); break; }
        int nk = ev_chance ? ev_n : mask_count(ev_legal, ev_n);
        if (!ev_chance) {
            long before = T->isets->n;
            long is = tab_find(T->isets, pl->ev_agent, pl->ev_obs, pl->ev_obs_len, ev_n,
                               ev_legal, 1);
            if (T->isets->n != before)
                memcpy(T->isets->ssum + T->isets->aoff[is], pl->ev_probs,
                       sizeof(double) * (size_t)ev_n);
            T->n[node].iset = is;
            T->n[node].agent = pl->ev_agent;
        }
        free(pl);
        if (T->nkids + nk > T->capkids) {
            T->capkids = T->capkids ? T->capkids * 2 : 4096;
            while (T->nkids + nk > T->capkids) T->capkids *= 2;
            T->kids = (long *)xrealloc(T->kids, sizeof(long) * (size_t)T->capkids);
            T->krew = (long *)xrealloc(T->krew, sizeof(long) * (size_t)T->capkids);
            T->kact = (int *)xrealloc(T->kact, sizeof(int) * (size_t)T->capkids);
        }
        long base = T->nkids;
        T->nkids += nk;
        T->n[node].nkid = nk;
        T->n[node].kid = base;
        int *child = (int *)xmalloc(sizeof(int) * (size_t)(nplan + 1));
        if (nplan) memcpy(child, plan, sizeof(int) * (size_t)nplan);
        int j = 0;
        for (int o = 0; o < ev_n; o++) {
            if (!ev_chance && !((ev_legal >> o) & 1u)) continue;
            child[nplan] = o;
            long pay = -1;
            long c = T->fail ? -1 : tree_arrive(T, snap, child, nplan + 1, in_init, &pay,
                                                depth + 1);
            T->kact[base + j] = o;
            T->kids[base + j] = c;
            T->krew[base + j] = pay;
            j++;
        }
        free(child);
        T->order[T->norder++] = node;
        break;
    }
    /* Each step-start state passed on the way leads to this node. */
    for (int b = 0; b < nb && !T->fail; b++) {
        double since[MAX_AGENTS];
        for (int a = 0; a < MAX_AGENTS; a++) since[a] = pend[a] - bpend[(size_t)b * MAX_AGENTS + a];
        long pay = tree_pay(T, since);
        double *m = T->memo->regret + T->memo->aoff[brow[b]];
        m[0] = (double)node;
        m[1] = (double)pay;
    }
    if (!T->fail) *pay_out = tree_pay(T, pend);
    free(brow); free(bpend); free(snap);
    return node;
}

static void tree_free(Tree *T) {
    if (!T) return;
    free(T->n); free(T->kids); free(T->krew); free(T->kact); free(T->rews); free(T->order);
    free(T->keybuf);
    tab_free(T->isets);
    tab_free(T->memo);
    free(T->vm->s); free(T->vm);
    free(T);
}

/* Follow everything, reading the policy's probabilities at every situation on
 * the way. On failure T->fail says why and the result is not to be used. */
static Tree *tree_build(const Program *p, const Policy *pol, int horizon, long max_nodes,
                        int merge) {
    Tree *T = (Tree *)xcalloc(1, sizeof(Tree));
    T->p = p;
    T->horizon = horizon;
    T->nag = p->nagents;
    T->merge = merge;
    T->max_nodes = max_nodes;
    for (int i = 0; i < p->ndecls; i++)
        if (p->decls[i].base + p->decls[i].len > T->nstate)
            T->nstate = p->decls[i].base + p->decls[i].len;
    T->keybuf = (double *)xmalloc(sizeof(double) * (size_t)(T->nstate + 1));
    T->isets = tab_new();
    T->memo = tab_new();
    T->vm = (VM *)xmalloc(sizeof(VM));
    State *s = (State *)xmalloc(sizeof(State));
    vm_init(T->vm, p, s);
    policy_attach(T->vm, pol);
    State root;
    state_reset(p, &root, 1);
    root.ep = 0;
    T->root = tree_arrive(T, &root, NULL, 0, 1, &T->root_rew, 0);
    if (T->root < 0 && !T->fail) T->fail = TREE_TOO_BIG;
    return T;
}

static const double *tree_probs(const Tree *T, const TNode *n) {
    return T->isets->ssum + T->isets->aoff[n->iset];
}

/* The chance of each edge out of a node. An agent whose bit is set in
 * `uniform` plays every legal action alike; one whose bit is set in `skip`
 * counts as 1 (the reach that is not its own doing). */
static double tree_edge(const Tree *T, const TNode *n, int j, uint32_t uniform, uint32_t skip) {
    if (n->kind == T_CHANCE) return 1.0 / (double)n->nkid;
    if ((skip >> n->agent) & 1u) return 1.0;
    if ((uniform >> n->agent) & 1u) return 1.0 / (double)n->nkid;
    return tree_probs(T, n)[T->kact[n->kid + j]];
}

/* What agent `a` is paid along edge e. */
static double tree_pays(const Tree *T, long e, int a) {
    long r = T->krew[e];
    return r < 0 ? 0.0 : T->rews[r + a];
}

static double tree_root_pays(const Tree *T, int a) {
    return T->root_rew < 0 ? 0.0 : T->rews[T->root_rew + a];
}

/* out[a]: what agent a collects per episode. With `keep`, also what each
 * agent collects from every node on (nag values a node), for the caller to
 * free. */
static void tree_eval_keep(const Tree *T, uint32_t uniform, double *out, double **keep) {
    int nag = T->nag;
    double *val = (double *)xcalloc((size_t)T->nn * (size_t)nag, sizeof(double));
    for (long k = 0; k < T->norder; k++) {
        long id = T->order[k];
        const TNode *n = &T->n[id];
        double *v = val + id * nag;
        for (int j = 0; j < n->nkid; j++) {
            long e = n->kid + j, c = T->kids[e];
            double pr = tree_edge(T, n, j, uniform, 0);
            if (pr == 0.0) continue;
            for (int a = 0; a < nag; a++) v[a] += pr * (tree_pays(T, e, a) + val[c * nag + a]);
        }
    }
    for (int a = 0; a < nag; a++) out[a] = tree_root_pays(T, a) + val[T->root * nag + a];
    if (keep) *keep = val;
    else free(val);
}

static void tree_eval(const Tree *T, uint32_t uniform, double *out) {
    tree_eval_keep(T, uniform, out, NULL);
}

/* reach[node]: how many times an episode passes through it, on average. */
static void tree_reach(const Tree *T, uint32_t uniform, uint32_t skip, double *reach) {
    memset(reach, 0, sizeof(double) * (size_t)T->nn);
    reach[T->root] = 1.0;
    for (long k = T->norder - 1; k >= 0; k--) {
        long id = T->order[k];
        const TNode *n = &T->n[id];
        if (reach[id] == 0.0) continue;
        for (int j = 0; j < n->nkid; j++)
            reach[T->kids[n->kid + j]] += reach[id] * tree_edge(T, n, j, uniform, skip);
    }
}

/* The best reply of one agent to everyone else's policy. */
typedef struct {
    const Tree *T;
    int me;
    double *reach;      /* how often each node is reached, the agent's own choices aside */
    double *val;        /* what the reply collects from each node on */
    char *have;
    int *pick;          /* per situation: the action chosen, -1 not yet, -2 being chosen */
    long *first, *next; /* the nodes of each situation */
    int all;            /* choose at every node by itself: the agent sees everything */
    int fixed;          /* the choices are made; only work out what they collect */
    int unsure;         /* a weight was too large to hold: the choice there is a guess */
} Reply;

static double reply_node(Reply *B, long id);

static int reply_first_legal(const Table *I, long is) {
    int a = 0;
    while (a < I->nact[is] - 1 && !((I->mask[is] >> a) & 1u)) a++;
    return a;
}

static int reply_pick(Reply *B, long is) {
    const Tree *T = B->T;
    const Table *I = T->isets;
    if (B->pick[is] >= 0) return B->pick[is];
    /* Being chosen further up this very path, or never reached: any action. */
    if (B->pick[is] == -2 || B->fixed) return reply_first_legal(I, is);
    B->pick[is] = -2;
    double q[MAX_ACT] = {0};
    /* The weight of a node counts every way of the agent's own to reach it,
     * and in a graph with merged nodes that count can pass what a double
     * holds. With one node the weight does not matter. With several, the
     * choice is then made on even weights and is not called a proof. */
    int one = B->first[is] >= 0 && B->next[B->first[is]] < 0, even = 0;
    for (long h = B->first[is]; h >= 0 && !one; h = B->next[h])
        if (!isfinite(B->reach[h])) even = 1;
    if (even) B->unsure = 1;
    for (long h = B->first[is]; h >= 0; h = B->next[h]) {
        const TNode *n = &T->n[h];
        double w = (one || even) ? 1.0 : B->reach[h];
        if (w == 0.0) continue;
        for (int j = 0; j < n->nkid; j++) {
            long e = n->kid + j;
            q[T->kact[e]] += w * (tree_pays(T, e, B->me) + reply_node(B, T->kids[e]));
        }
    }
    int best = reply_first_legal(I, is);
    for (int a = 0; a < I->nact[is]; a++)
        if (((I->mask[is] >> a) & 1u) && q[a] > q[best]) best = a;
    B->pick[is] = best;
    return best;
}

static double reply_node(Reply *B, long id) {
    if (B->have[id]) return B->val[id];
    const Tree *T = B->T;
    const TNode *n = &T->n[id];
    double v = 0.0;
    if (n->kind == T_DECIDE && n->agent == B->me) {
        if (B->all) {
            v = -1e300;
            for (int j = 0; j < n->nkid; j++) {
                long e = n->kid + j;
                double x = tree_pays(T, e, B->me) + reply_node(B, T->kids[e]);
                if (x > v) v = x;
            }
        } else {
            int a = reply_pick(B, n->iset);
            for (int j = 0; j < n->nkid; j++) {
                long e = n->kid + j;
                if (T->kact[e] == a) v = tree_pays(T, e, B->me) + reply_node(B, T->kids[e]);
            }
        }
    } else {
        for (int j = 0; j < n->nkid; j++) {
            long e = n->kid + j;
            double pr = tree_edge(T, n, j, 0, 0);
            if (pr != 0.0) v += pr * (tree_pays(T, e, B->me) + reply_node(B, T->kids[e]));
        }
    }
    /* While a situation is still being chosen, values under it are provisional. */
    B->val[id] = v;
    B->have[id] = 1;
    return v;
}

/* What `me` collects by replying as well as it can to the others' policy.
 *
 * The reply is one action for each thing the agent can be shown, and the
 * figure returned is what that reply really collects: it is worked out again
 * from the choices once they are made. So it is always a return some policy
 * of the agent reaches. It is the most any policy can reach when *sure is 1,
 * which is when either holds: the agent is never shown the same thing after
 * two different histories of its own (it remembers), or each thing it can be
 * shown comes up at one node only (it sees all that matters). Otherwise the
 * best may be higher.
 *
 * With `all`, the agent chooses seeing the whole state, which is the most
 * anything could collect. */
static double tree_reply(const Tree *T, int me, int all, int *sure, int *pick_out) {
    Reply B;
    memset(&B, 0, sizeof(B));
    B.T = T;
    B.me = me;
    B.all = all;
    long ni = T->isets->n, nn = T->nn;
    B.reach = (double *)xcalloc((size_t)nn, sizeof(double));
    B.val = (double *)xcalloc((size_t)nn, sizeof(double));
    B.have = (char *)xcalloc((size_t)nn, 1);
    B.pick = (int *)xmalloc(sizeof(int) * (size_t)(ni ? ni : 1));
    B.first = (long *)xmalloc(sizeof(long) * (size_t)(ni ? ni : 1));
    B.next = (long *)xmalloc(sizeof(long) * (size_t)nn);
    for (long i = 0; i < ni; i++) { B.pick[i] = -1; B.first[i] = -1; }
    tree_reach(T, 0, (uint32_t)1 << me, B.reach);

    /* seq[node]: the agent's own situations and actions on the way there, as
     * one number. A node reached by two different ones is marked `mixed`. */
    uint64_t *seq = (uint64_t *)xcalloc((size_t)nn, sizeof(uint64_t));
    uint64_t *iseq = (uint64_t *)xcalloc((size_t)(ni ? ni : 1), sizeof(uint64_t));
    char *seen = (char *)xcalloc((size_t)nn, 1), *mixed = (char *)xcalloc((size_t)nn, 1);
    int ok = 1, single = 1;
    seen[T->root] = 1;
    for (long k = T->norder - 1; k >= 0; k--) {
        long id = T->order[k];
        const TNode *n = &T->n[id];
        if (!seen[id]) continue;
        int mine = n->kind == T_DECIDE && n->agent == me;
        if (mine) {
            if (mixed[id]) ok = 0;
            if (B.first[n->iset] < 0) iseq[n->iset] = seq[id];
            else {
                single = 0;
                if (iseq[n->iset] != seq[id]) ok = 0;
            }
            B.next[id] = B.first[n->iset];
            B.first[n->iset] = id;
        }
        for (int j = 0; j < n->nkid; j++) {
            long e = n->kid + j, c = T->kids[e];
            uint64_t sc = mine ? mix64(mix64(seq[id] + 1, (uint64_t)n->iset), (uint64_t)T->kact[e])
                               : seq[id];
            if (!seen[c]) { seen[c] = 1; seq[c] = sc; mixed[c] = mixed[id]; }
            else if (seq[c] != sc || mixed[id]) mixed[c] = 1;
        }
    }
    reply_node(&B, T->root);
    if (!all) {
        B.fixed = 1;
        memset(B.have, 0, (size_t)nn);
    }
    double v = tree_root_pays(T, me) + reply_node(&B, T->root);
    if (sure) *sure = (ok || single) && !B.unsure;
    if (pick_out) memcpy(pick_out, B.pick, sizeof(int) * (size_t)ni);
    free(B.reach); free(B.val); free(B.have); free(B.pick); free(B.first); free(B.next);
    free(seq); free(iseq); free(seen); free(mixed);
    return v;
}

/* ------------------------------------------------------------------ */
/* Sampled evaluation                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    double mean[MAX_AGENTS], sem[MAX_AGENTS];
    long decisions, unseen;     /* decisions a table was asked, and had no row for */
} Eval;

static void eval_any(const Program *p, const Policy *pol, uint32_t random_agents,
                     Driver *driver, int episodes, uint64_t seed, int horizon, Eval *out) {
    State s;
    VM vm;
    vm_init(&vm, p, &s);
    policy_attach(&vm, pol);
    vm.random_agents = random_agents;
    vm.driver = driver;
    double sum[MAX_AGENTS] = {0}, sq[MAX_AGENTS] = {0}, ret[MAX_AGENTS];
    for (int e = 0; e < episodes; e++) {
        run_episode(&vm, NULL, NULL, seed + (uint64_t)e * 7919ULL, 0, e, ret, horizon);
        for (int a = 0; a < p->nagents; a++) { sum[a] += ret[a]; sq[a] += ret[a] * ret[a]; }
    }
    memset(out, 0, sizeof(*out));
    for (int a = 0; a < p->nagents; a++) {
        double m = sum[a] / (double)episodes;
        double var = sq[a] / (double)episodes - m * m;
        out->mean[a] = m;
        out->sem[a] = episodes > 1 && var > 0 ? sqrt(var / (double)(episodes - 1)) : 0.0;
    }
    out->decisions = vm.tab_seen;
    out->unseen = vm.tab_missed;
}

/* Counts what a policy does in each situation it meets, for the report. */
typedef struct {
    Driver dr;
    Table *seen;        /* regret: how often each action was taken; ssum: the policy there */
    long cap;           /* stop listing past this many situations */
    int full;
    double *freq;       /* [agent][action]: how often each action was taken in all */
} Census;

static void census_decided(Driver *dr, VM *vm, int agent, int n_act, uint32_t legal,
                           const double *probs, int action) {
    Census *C = (Census *)dr;
    C->freq[agent * MAX_ACT + action] += 1.0;
    if (C->full) return;
    long before = C->seen->n;
    long id = tab_find(C->seen, agent, vm->scratch, vm->p->agents[agent].obs_dim, n_act,
                       legal, 1);
    if (C->seen->n != before)
        memcpy(C->seen->ssum + C->seen->aoff[id], probs, sizeof(double) * (size_t)n_act);
    C->seen->regret[C->seen->aoff[id] + action] += 1.0;
    if (C->seen->n >= C->cap) C->full = 1;
}

/* ------------------------------------------------------------------ */
/* Report: what a policy does and what it is worth                     */
/* ------------------------------------------------------------------ */

typedef struct { double share; long id; } RowRank;

static int cmp_rank_desc(const void *a, const void *b) {
    double x = ((const RowRank *)a)->share, y = ((const RowRank *)b)->share;
    return x < y ? 1 : x > y ? -1 : 0;
}

/* A number that reads back as the same double: what a policy row is keyed on. */
static void json_exact(FILE *f, double v) {
    if (!(v == v) || v > 1e300 || v < -1e300) { fputs("0", f); return; }
    if (v == (double)(long long)v && fabs(v) < 1e15) fprintf(f, "%lld", (long long)v);
    else fprintf(f, "%.17g", v);
}

static void json_vec(FILE *f, const double *v, int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) fprintf(f, "%s%.12g", i ? "," : "", v[i]);
    fputc(']', f);
}

/* The situations a policy meets most, each with what it does there. `share`
 * is how many times an episode comes to it, on average. */
static void report_rows(FILE *f, const Table *I, const double *share, const double *q,
                        int max_rows) {
    RowRank *rank = (RowRank *)xmalloc(sizeof(RowRank) * (size_t)(I->n ? I->n : 1));
    for (long i = 0; i < I->n; i++) { rank[i].share = share[i]; rank[i].id = i; }
    qsort(rank, (size_t)I->n, sizeof(RowRank), cmp_rank_desc);
    /* Up to max_rows for each agent, so one agent's common situations never
     * crowd another's out of the list. */
    long per_agent[MAX_AGENTS] = {0}, listed[MAX_AGENTS] = {0};
    for (long i = 0; i < I->n; i++) per_agent[I->agent[i]]++;
    fprintf(f, "\"situations\":%ld,\"situations_by_agent\":[", I->n);
    for (int g = 0; g < MAX_AGENTS; g++) fprintf(f, "%s%ld", g ? "," : "", per_agent[g]);
    fputs("],\"rows\":[", f);
    int any = 0;
    for (long k = 0; k < I->n; k++) {
        long id = rank[k].id;
        if (listed[I->agent[id]] >= max_rows) continue;
        listed[I->agent[id]]++;
        double probs[MAX_ACT];
        tab_average(I, id, probs);
        fprintf(f, "%s{\"agent\":%d,\"legal\":%u,\"share\":%.9g,\"obs\":", any++ ? "," : "",
                I->agent[id], I->mask[id], rank[k].share);
        fputc('[', f);
        for (int i = 0; i < I->klen[id]; i++) {
            if (i) fputc(',', f);
            json_exact(f, I->keys[I->koff[id] + i]);
        }
        fputs("],\"probs\":", f);
        json_vec(f, probs, I->nact[id]);
        if (q && share[id] > 0.0) {
            /* What the agent goes on to collect after each action here, over
             * every instant in which it is shown this, weighed by how often
             * the policies in play bring each about. */
            fputs(",\"value\":[", f);
            for (int a = 0; a < I->nact[id]; a++) {
                if (a) fputc(',', f);
                if ((I->mask[id] >> a) & 1u) fprintf(f, "%.12g", q[I->aoff[id] + a] / share[id]);
                else fputs("null", f);
            }
            fputc(']', f);
        }
        fputc('}', f);
    }
    fputs("],", f);
    free(rank);
}

/* Counts the actions taken, for the report of a search. */
typedef struct { Driver dr; double freq[MAX_AGENTS * MAX_ACT]; } Tally;

static void tally_decided(Driver *dr, VM *vm, int agent, int n_act, uint32_t legal,
                          const double *probs, int action) {
    (void)vm; (void)n_act; (void)legal; (void)probs;
    ((Tally *)dr)->freq[agent * MAX_ACT + action] += 1.0;
}

/* Episodes on the stream eval_any uses, until `max_episodes` have been played
 * or the time is `until`: at least one, and one that has begun is finished.
 * Returns how many. */
static int eval_until(const Program *p, const Policy *pol, uint32_t random_agents,
                      Driver *driver, int max_episodes, double until, uint64_t seed,
                      int horizon, Eval *out) {
    State *s = (State *)xmalloc(sizeof(State));
    VM *vm = (VM *)xmalloc(sizeof(VM));
    vm_init(vm, p, s);
    policy_attach(vm, pol);
    vm->random_agents = random_agents;
    vm->driver = driver;
    double sum[MAX_AGENTS] = {0}, sq[MAX_AGENTS] = {0}, ret[MAX_AGENTS];
    int n = 0;
    while (n < max_episodes && (n == 0 || now_sec() < until)) {
        run_episode(vm, NULL, NULL, seed + (uint64_t)n * 7919ULL, 0, n, ret, horizon);
        for (int a = 0; a < p->nagents; a++) { sum[a] += ret[a]; sq[a] += ret[a] * ret[a]; }
        n++;
    }
    memset(out, 0, sizeof(*out));
    for (int a = 0; a < p->nagents; a++) {
        double m = sum[a] / (double)n;
        double var = sq[a] / (double)n - m * m;
        out->mean[a] = m;
        out->sem[a] = n > 1 && var > 0 ? sqrt(var / (double)(n - 1)) : 0.0;
    }
    free(vm); free(s);
    return n;
}

/* How a search's network was made, or null when it has none. */
static void json_guide(FILE *f, const Guide *G) {
    if (!G) { fputs("null", f); return; }
    fprintf(f, "{\"games\":%ld,\"decisions\":%ld,\"rounds\":%ld,\"sec\":%.3f,\"cores\":%d,"
            "\"selfplay_playouts\":%d,\"width\":%d,\"inputs\":%d}", G->games, G->samples,
            G->rounds, G->sec, G->cores, G->playouts, G->net[0]->hid, G->net[0]->nin);
}

/* A search cannot be weighed over every outcome: each of its decisions is
 * thousands of playouts, and it has no table to read. It is sampled, and
 * since an episode of it takes seconds, for a set time (--score-sec, 15 s
 * unless said): a share for every agent searching, and a share for each
 * agent searching against the others playing uniformly at random. Every
 * measurement plays at least one episode and finishes the one it is in, so a
 * world with long episodes overruns. */
static int report_search(const Program *p, const Policy *pol, int horizon, uint64_t seed,
                         int argc, char **argv) {
    const Search *S = pol->search;
    int nag = p->nagents;
    int episodes = (int)arg_long(argc, argv, "--episodes", 4000);
    double seconds = arg_double(argc, argv, "--score-sec", 15.0);
    if (episodes < 1) episodes = 1;
    FILE *f = stdout;
    double t0 = now_sec();
    int parts = nag > 1 ? nag + 1 : 1;
    Tally tally;
    memset(&tally, 0, sizeof(tally));
    tally.dr.decided = tally_decided;
    Eval ev, rnd, vs;
    eval_any(p, NULL, mask_all(nag), NULL, 4000, seed, horizon, &rnd);
    /* Each measurement ends at its own mark, so one that overruns takes the
     * time from the next and the whole stays near what was asked. With no
     * limit (0) the episode count alone decides. */
    double never = t0 + 1e9;
    int n_all = eval_until(p, pol, 0, &tally.dr, episodes,
                           seconds > 0 ? t0 + seconds / parts : never, seed, horizon, &ev);
    double vsm[MAX_AGENTS] = {0}, vss[MAX_AGENTS] = {0};
    int vsn[MAX_AGENTS] = {0};
    for (int a = 0; a < nag && nag > 1; a++) {
        vsn[a] = eval_until(p, pol, mask_all(nag) & ~((uint32_t)1 << a), NULL, episodes,
                            seconds > 0 ? t0 + seconds * (a + 2) / parts : never, seed,
                            horizon, &vs);
        vsm[a] = vs.mean[a];
        vss[a] = vs.sem[a];
    }
    fputs("{\"world\":", f);
    json_str(f, p->world);
    fprintf(f, ",\"horizon\":%d,\"exact\":false,\"why\":\"search\",\"limit\":0,", horizon);
    fprintf(f, "\"search\":{\"playouts\":%d,\"explore\":%.9g,\"hidden\":\"%s\","
            "\"positions\":%ld,\"repeats\":%ld,\"distinct\":%ld,\"conflicts\":%ld,"
            "\"guide\":", S->playouts, S->explore, HIDDEN_NAME[S->hidden], S->checked[0],
            S->checked[1], S->checked[2], S->checked[3]);
    json_guide(f, S->guide);
    fprintf(f, ",\"rollout\":%.9g},", S->guide ? S->mix : 1.0);
    fprintf(f, "\"episodes\":%d,\"score_sec\":%.9g,", n_all, seconds);
    fputs("\"return\":", f); json_vec(f, ev.mean, nag);
    fputs(",\"sem\":", f); json_vec(f, ev.sem, nag);
    fputs(",\"random\":", f); json_vec(f, rnd.mean, nag);
    fputs(",\"random_sem\":", f); json_vec(f, rnd.sem, nag);
    fputs(",\"random_episodes\":4000", f);
    fputs(",\"vs_random\":", f); json_vec(f, vsm, nag);
    fputs(",\"vs_random_sem\":", f); json_vec(f, vss, nag);
    fputs(",\"vs_random_episodes\":[", f);
    for (int a = 0; a < nag; a++) fprintf(f, "%s%d", a ? "," : "", vsn[a]);
    /* A search keeps no table, so there are no situations to list. */
    fputs("],\"decisions\":0,\"unseen\":0,\"listed_all\":true,\"situations\":0,"
          "\"situations_by_agent\":[", f);
    for (int g = 0; g < MAX_AGENTS; g++) fprintf(f, "%s0", g ? "," : "");
    fputs("],\"rows\":[],\"freq\":[", f);
    for (int a = 0; a < nag; a++) {
        if (a) fputc(',', f);
        for (int k = 0; k < MAX_ACT; k++) tally.freq[a * MAX_ACT + k] /= (double)n_all;
        json_vec(f, tally.freq + a * MAX_ACT, p->agents[a].n_act);
    }
    fprintf(f, "],\"sec\":%.3f}\n", now_sec() - t0);
    return 0;
}

static int cmd_report(const Program *p, const Policy *pol, int horizon, uint64_t seed,
                      int argc, char **argv) {
    if (pol->search) return report_search(p, pol, horizon, seed, argc, argv);
    long max_nodes = arg_long(argc, argv, "--exact-nodes", 2000000);
    int episodes = (int)arg_long(argc, argv, "--episodes", 4000);
    int max_rows = (int)arg_long(argc, argv, "--rows", 400);
    int nag = p->nagents;
    FILE *f = stdout;
    double t0 = now_sec();
    Tree *T = max_nodes > 0
        ? tree_build(p, pol, horizon, max_nodes, !has_flag(argc, argv, "--no-merge")) : NULL;

    fputs("{\"world\":", f);
    json_str(f, p->world);
    fprintf(f, ",\"horizon\":%d,", horizon);

    if (T && !T->fail) {
        double ret[MAX_AGENTS], rnd[MAX_AGENTS], vs[MAX_AGENTS], tmp[MAX_AGENTS];
        double best[MAX_AGENTS];
        int sure[MAX_AGENTS];
        double *val = NULL;
        tree_eval_keep(T, 0, ret, &val);
        fprintf(f, "\"exact\":true,\"positions\":%ld,", T->nn);
        fputs("\"return\":", f); json_vec(f, ret, nag);
        /* --values: only what each action is worth in each situation, for
         * `hunch why`; the comparisons below are the slow part. */
        if (!has_flag(argc, argv, "--values")) {
            tree_eval(T, mask_all(nag), rnd);
            for (int a = 0; a < nag; a++) {
                tree_eval(T, mask_all(nag) & ~((uint32_t)1 << a), tmp);
                vs[a] = tmp[a];
                best[a] = tree_reply(T, a, 0, &sure[a], NULL);
                /* A reply that collects what the agent would by seeing the
                 * whole state is the best there is, however it was found. */
                if (!sure[a] && nag > 1 && tree_reply(T, a, 1, NULL, NULL) - best[a] < 1e-9)
                    sure[a] = 1;
            }
            double top1 = nag == 1 ? tree_reply(T, 0, 1, NULL, NULL) : 0.0;
            if (nag == 1 && top1 - best[0] < 1e-9) sure[0] = 1;
            fputs(",\"random\":", f); json_vec(f, rnd, nag);
            fputs(",\"vs_random\":", f); json_vec(f, vs, nag);
            fputs(",\"best_reply\":", f); json_vec(f, best, nag);
            fputs(",\"sure\":[", f);
            for (int a = 0; a < nag; a++)
                fprintf(f, "%s%s", a ? "," : "", sure[a] ? "true" : "false");
            fputc(']', f);
            /* A policy that collects what seeing everything would is the best. */
            if (nag == 1) fprintf(f, ",\"seeing_all\":%.12g", top1);
        }
        fputc(',', f);
        /* how often each situation comes up, and each action is taken */
        double *reach = (double *)xcalloc((size_t)T->nn, sizeof(double));
        double *share = (double *)xcalloc((size_t)(T->isets->n ? T->isets->n : 1), sizeof(double));
        double *freq = (double *)xcalloc((size_t)MAX_AGENTS * MAX_ACT, sizeof(double));
        long nq = 1;
        for (long i = 0; i < T->isets->n; i++) nq += T->isets->nact[i];
        double *q = (double *)xcalloc((size_t)nq, sizeof(double));
        tree_reach(T, 0, 0, reach);
        for (long id = 0; id < T->nn; id++) {
            const TNode *n = &T->n[id];
            if (n->kind != T_DECIDE) continue;
            share[n->iset] += reach[id];
            const double *pr = tree_probs(T, n);
            for (int j = 0; j < n->nkid; j++) {
                long e = n->kid + j;
                int act = T->kact[e];
                freq[n->agent * MAX_ACT + act] += reach[id] * pr[act];
                q[T->isets->aoff[n->iset] + act] +=
                    reach[id] * (tree_pays(T, e, n->agent) + val[T->kids[e] * nag + n->agent]);
            }
        }
        report_rows(f, T->isets, share, q, max_rows);
        free(q); free(val);
        fputs("\"freq\":[", f);
        for (int a = 0; a < nag; a++) {
            if (a) fputc(',', f);
            json_vec(f, freq + a * MAX_ACT, p->agents[a].n_act);
        }
        fputs("],", f);
        free(reach); free(share); free(freq);
    } else {
        fputs("\"exact\":false,\"why\":", f);
        if (!T) json_str(f, "off");
        else if (T->fail == TREE_RAND) {
            char why[96];
            snprintf(why, sizeof(why), "rand:%d:%d", T->rand_line, T->rand_col);
            json_str(f, why);
        } else json_str(f, "big");
        fprintf(f, ",\"limit\":%ld,\"episodes\":%d,", max_nodes, episodes);
        Census C;
        memset(&C, 0, sizeof(C));
        C.dr.decided = census_decided;
        C.seen = tab_new();
        C.cap = 20000;
        C.freq = (double *)xcalloc((size_t)MAX_AGENTS * MAX_ACT, sizeof(double));
        Eval ev, rnd, vs;
        eval_any(p, pol, 0, &C.dr, episodes, seed, horizon, &ev);
        eval_any(p, pol, mask_all(nag), NULL, episodes, seed, horizon, &rnd);
        fputs("\"return\":", f); json_vec(f, ev.mean, nag);
        fputs(",\"sem\":", f); json_vec(f, ev.sem, nag);
        fputs(",\"random\":", f); json_vec(f, rnd.mean, nag);
        fputs(",\"random_sem\":", f); json_vec(f, rnd.sem, nag);
        double vsm[MAX_AGENTS] = {0}, vss[MAX_AGENTS] = {0};
        for (int a = 0; a < nag && nag > 1; a++) {
            eval_any(p, pol, mask_all(nag) & ~((uint32_t)1 << a), NULL, episodes, seed,
                     horizon, &vs);
            vsm[a] = vs.mean[a];
            vss[a] = vs.sem[a];
        }
        fputs(",\"vs_random\":", f); json_vec(f, vsm, nag);
        fputs(",\"vs_random_sem\":", f); json_vec(f, vss, nag);
        fprintf(f, ",\"decisions\":%ld,\"unseen\":%ld,\"listed_all\":%s,", ev.decisions,
                ev.unseen, C.full ? "false" : "true");
        double *share = (double *)xcalloc((size_t)(C.seen->n ? C.seen->n : 1), sizeof(double));
        for (long id = 0; id < C.seen->n; id++) {
            for (int a = 0; a < C.seen->nact[id]; a++)
                share[id] += C.seen->regret[C.seen->aoff[id] + a];
            share[id] /= (double)episodes;
        }
        report_rows(f, C.seen, share, NULL, max_rows);
        fputs("\"freq\":[", f);
        for (int a = 0; a < nag; a++) {
            if (a) fputc(',', f);
            for (int k = 0; k < MAX_ACT; k++) C.freq[a * MAX_ACT + k] /= (double)episodes;
            json_vec(f, C.freq + a * MAX_ACT, p->agents[a].n_act);
        }
        fputs("],", f);
        free(share); free(C.freq);
        tab_free(C.seen);
    }
    fprintf(f, "\"sec\":%.3f}\n", now_sec() - t0);
    tree_free(T);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Why: every decision of one episode, replayed once per action        */
/* ------------------------------------------------------------------ */

/* The episode is run once. Then each decision in it is gone back to: the
 * state is restored to that instant, each legal action is forced in turn, and
 * the episode is played on to its end. `samples` continuations are tried, the
 * same ones for every action, so two actions are compared on the same luck.
 * The first continuation is the one that really happened. */
/* Why, for a search. Replaying each decision once per action, hundreds of
 * times, with a search in every seat of every continuation would take hours.
 * The search has already done the comparing: the episode is played once, and
 * for each decision this says how many playouts tried each action and the
 * mean of what its agent collected in them, from that instant to the end. In
 * a playout the first moves are the search's and the rest are random, so a
 * mean is the search's estimate, not what the action is worth against it. */
static int why_search(const Program *p, const Policy *pol, int horizon, uint64_t seed,
                      int only) {
    State *s = (State *)xmalloc(sizeof(State));
    VM *vm = (VM *)xmalloc(sizeof(VM));
    vm_init(vm, p, s);
    policy_attach(vm, pol);
    state_reset(p, s, seed);
    s->ep = 0;
    vm_run(vm, p->init_entry);
    memset(vm->agent_reward, 0, sizeof(vm->agent_reward));
    FILE *f = stdout;
    double total[MAX_AGENTS] = {0};
    fputs("{\"world\":", f);
    json_str(f, p->world);
    fprintf(f, ",\"seed\":%llu,\"search\":{\"playouts\":%d,\"explore\":%.9g,"
            "\"hidden\":\"%s\",\"guide\":", (unsigned long long)seed,
            pol->search->playouts, pol->search->explore, HIDDEN_NAME[pol->search->hidden]);
    json_guide(f, pol->search->guide);
    fprintf(f, ",\"rollout\":%.9g},\"decisions\":[", pol->search->guide ? pol->search->mix : 1.0);
    int first = 1;
    for (int t = 0; t < horizon && !s->done; t++) {
        vm->record = 1;
        vm_step(vm);
        vm->record = 0;
        for (int a = 0; a < p->nagents; a++) total[a] += vm->agent_reward[a];
        if (only > 0 && only != t + 1) continue;
        for (int k = 0; k < vm->n_dec; k++) {
            const Decision *d = &vm->dec[k];
            fprintf(f, "%s{\"step\":%d,\"index\":%d,\"agent\":%d,\"action\":%d,\"legal\":%u,"
                    "\"obs\":", first ? "" : ",", t + 1, k, d->agent, d->action, d->mask);
            first = 0;
            fputc('[', f);
            for (int i = 0; i < d->obs_dim; i++) {
                if (i) fputc(',', f);
                json_num(f, d->obs[i]);
            }
            fputs("],\"probs\":", f);
            json_vec(f, d->prob, d->n_act);
            fputs(",\"tries\":[", f);
            for (int a = 0; a < d->n_act; a++) fprintf(f, "%s%d", a ? "," : "", d->tries[a]);
            /* null: illegal, or legal and never tried (fewer playouts than actions) */
            fputs("],\"value\":[", f);
            for (int a = 0; a < d->n_act; a++) {
                if (a) fputc(',', f);
                if (d->tries[a] > 0) fprintf(f, "%.9g", d->mean[a]);
                else fputs("null", f);
            }
            fputs("]}", f);
        }
    }
    fputs("],\"returns\":", f);
    json_vec(f, total, p->nagents);
    fputs("}\n", f);
    free(vm); free(s);
    return 0;
}

static int cmd_why(const Program *p, const Policy *pol, int horizon, uint64_t seed,
                   int argc, char **argv) {
    int samples = (int)arg_long(argc, argv, "--samples", 200);
    int only = (int)arg_long(argc, argv, "--step", 0);
    if (pol->search) return why_search(p, pol, horizon, seed, only);
    if (samples < 1) samples = 1;
    State s, bs;
    VM vm, bvm;
    vm_init(&vm, p, &s);
    vm_init(&bvm, p, &bs);
    policy_attach(&vm, pol);
    policy_attach(&bvm, pol);
    state_reset(p, &s, seed);
    s.ep = 0;
    vm_run(&vm, p->init_entry);
    memset(vm.agent_reward, 0, sizeof(vm.agent_reward));

    FILE *f = stdout;
    double total[MAX_AGENTS] = {0};
    double *vals = (double *)xmalloc(sizeof(double) * (size_t)MAX_ACT * (size_t)samples);
    fputs("{\"world\":", f);
    json_str(f, p->world);
    fprintf(f, ",\"seed\":%llu,\"samples\":%d,\"decisions\":[", (unsigned long long)seed, samples);
    int first = 1;
    for (int t = 0; t < horizon && !s.done; t++) {
        State snap = s;
        vm.record = 1;
        vm_step(&vm);
        vm.record = 0;
        for (int a = 0; a < p->nagents; a++) total[a] += vm.agent_reward[a];
        if (only > 0 && only != t + 1) continue;
        Decision dec[MAX_DEC_PER_STEP];
        int nd = vm.n_dec;
        memcpy(dec, vm.dec, sizeof(Decision) * (size_t)nd);
        int plan[MAX_DEC_PER_STEP];
        for (int k = 0; k < nd; k++) {
            const Decision *d = &dec[k];
            for (int j = 0; j < k; j++) plan[j] = dec[j].action;
            for (int a = 0; a < d->n_act; a++) {
                if (!((d->mask >> a) & 1u)) continue;
                plan[k] = a;
                for (int j = 0; j < samples; j++) {
                    Plan pl;
                    plan_init(&pl, plan, k + 1, STOP_NEVER, -1, 0);
                    pl.reseed_at = k;
                    pl.reseed = j ? mix64(mix64(seed + 0x51ED, (uint64_t)(t * 64 + k)),
                                          (uint64_t)j) | 1u : 0;
                    plan_run(&bvm, &pl, &snap, 0);
                    double v = bvm.agent_reward[d->agent];
                    vals[a * samples + j] = v + play_out(&bvm, d->agent, horizon);
                }
            }
            fprintf(f, "%s{\"step\":%d,\"index\":%d,\"agent\":%d,\"action\":%d,\"legal\":%u,"
                    "\"obs\":", first ? "" : ",", t + 1, k, d->agent, d->action, d->mask);
            first = 0;
            fputc('[', f);
            for (int i = 0; i < d->obs_dim; i++) {
                if (i) fputc(',', f);
                json_num(f, d->obs[i]);
            }
            fputs("],\"probs\":", f);
            json_vec(f, d->prob, d->n_act);
            fputs(",\"value\":[", f);
            for (int a = 0; a < d->n_act; a++) {
                if (a) fputc(',', f);
                if (!((d->mask >> a) & 1u)) { fputs("null", f); continue; }
                double m = 0.0;
                for (int j = 0; j < samples; j++) m += vals[a * samples + j];
                fprintf(f, "%.9g", m / (double)samples);
            }
            fputs("],\"actual\":[", f);
            for (int a = 0; a < d->n_act; a++) {
                if (a) fputc(',', f);
                if (!((d->mask >> a) & 1u)) fputs("null", f);
                else fprintf(f, "%.9g", vals[a * samples]);
            }
            /* against the action taken, on the same continuations */
            fputs("],\"diff\":[", f);
            double se[MAX_ACT];
            for (int a = 0; a < d->n_act; a++) {
                if (a) fputc(',', f);
                se[a] = 0.0;
                if (!((d->mask >> a) & 1u)) { fputs("null", f); continue; }
                double m = 0.0, sq = 0.0;
                for (int j = 0; j < samples; j++) {
                    double x = vals[a * samples + j] - vals[d->action * samples + j];
                    m += x;
                    sq += x * x;
                }
                m /= (double)samples;
                double var = sq / (double)samples - m * m;
                se[a] = samples > 1 && var > 0 ? sqrt(var / (double)(samples - 1)) : 0.0;
                fprintf(f, "%.9g", m);
            }
            fputs("],\"se\":", f);
            json_vec(f, se, d->n_act);
            fputc('}', f);
        }
    }
    fputs("],\"returns\":", f);
    json_vec(f, total, p->nagents);
    fputs("}\n", f);
    free(vals);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Regret: self-play by regret matching over forked action values      */
/* ------------------------------------------------------------------ */

/* One agent at a time is the learner. An episode is played with everyone on
 * their current strategy, and at each decision of the learner the instant is
 * forked: every legal action is forced in turn and played on, which gives the
 * value of each. The regret of an action is its value less the value of the
 * strategy, and the next strategy plays each action in proportion to its
 * regret so far. In a two-player zero-sum game where each agent can tell
 * apart everything it has seen and done, the average of the strategies played
 * approaches an equilibrium.
 *
 * This is Monte Carlo counterfactual regret minimisation (Lanctot et al.,
 * 2009). For the learner's first `branch` decisions of an episode every
 * action is followed in full (external sampling). Past that, one action is
 * followed and the others are measured by a single play-out each (probing,
 * Gibson et al., 2012), so a long episode costs about its length times its
 * actions instead of growing with every decision. */
typedef struct {
    VM *vm;
    Table *tab;
    int horizon, me, branch;
    double explore;
    int plus;           /* never let a regret fall below zero */
    uint64_t rng[4];
} Regret;

static double regret_probe(Regret *R, const State *snap, const int *plan, int nplan,
                           int in_init) {
    Plan pl;
    plan_init(&pl, plan, nplan, STOP_NEVER, -1, 0);
    plan_run(R->vm, &pl, snap, in_init);
    double v = R->vm->agent_reward[R->me];
    return v + play_out(R->vm, R->me, R->horizon);
}

/* What the learner collects from `from` on, as sampled. `w` undoes the odds
 * of the learner's own sampled actions on the way here; `own` counts its
 * decisions so far. */
static double regret_walk(Regret *R, const State *from, const int *plan, int nplan,
                          double w, int own, int in_init) {
    VM *vm = R->vm;
    State *snap = (State *)xmalloc(sizeof(State));
    *snap = *from;
    double total = 0.0;
    Plan pl;
    for (;;) {
        plan_init(&pl, plan, nplan, STOP_AGENT, R->me, 0);
        pl.avg_weight = w;
        int r = plan_run(vm, &pl, snap, in_init);
        if (r == 1) {
            total += vm->agent_reward[R->me];
            *snap = *vm->s;
            plan = NULL;
            nplan = 0;
            in_init = 0;
            if (snap->done || snap->step >= R->horizon) break;
            continue;
        }
        long row = pl.ev_row;
        int n = pl.ev_n, nt = pl.ntrail;
        uint32_t legal = pl.ev_legal;
        double sigma[MAX_ACT], v[MAX_ACT];
        memcpy(sigma, pl.ev_probs, sizeof(double) * (size_t)n);
        int *child = (int *)xmalloc(sizeof(int) * (size_t)(nt + 1));
        memcpy(child, pl.trail, sizeof(int) * (size_t)nt);
        if (own < R->branch) {
            for (int a = 0; a < n; a++) {
                if (!((legal >> a) & 1u)) continue;
                child[nt] = a;
                v[a] = regret_walk(R, snap, child, nt + 1, w, own + 1, in_init);
            }
        } else {
            int nl = mask_count(legal, n), star = -1;
            double xi[MAX_ACT], acc = 0.0, u = rng_double(R->rng);
            for (int a = 0; a < n; a++) {
                xi[a] = ((legal >> a) & 1u)
                    ? R->explore / (double)nl + (1.0 - R->explore) * sigma[a] : 0.0;
                if (xi[a] > 0.0) {
                    star = a;
                    acc += xi[a];
                    if (u < acc) break;
                }
            }
            for (int a = 0; a < n; a++) {
                if (!((legal >> a) & 1u)) continue;
                child[nt] = a;
                v[a] = a == star
                    ? regret_walk(R, snap, child, nt + 1, w / xi[a], own + 1, in_init)
                    : regret_probe(R, snap, child, nt + 1, in_init);
            }
        }
        double ev = 0.0;
        for (int a = 0; a < n; a++)
            if ((legal >> a) & 1u) ev += sigma[a] * v[a];
        /* The table may have grown, and moved, while the branches ran. */
        double *reg = R->tab->regret + R->tab->aoff[row];
        for (int a = 0; a < n; a++) {
            if (!((legal >> a) & 1u)) continue;
            reg[a] += w * (v[a] - ev);
            if (R->plus && reg[a] < 0.0) reg[a] = 0.0;
        }
        total += ev;
        free(child);
        break;
    }
    free(snap);
    return total;
}

static double eval_mean0(const Program *p, const Policy *pol, int episodes, int horizon) {
    Eval ev;
    eval_any(p, pol, 0, NULL, episodes, EVAL_SEED, horizon, &ev);
    return ev.mean[0];
}

/* Returns the steps used. *iters_out gets the number of episodes each agent
 * learned from. */
static double train_regret(const Program *p, const Cfg *cfg, uint64_t seed, Table *tab,
                           Curve *curve, long *iters_out, int branch, double explore,
                           long max_rows, int linear, int plus) {
    State s;
    VM vm;
    vm_init(&vm, p, &s);
    vm.tab = tab;
    vm.tab_mode = TAB_CURRENT;
    vm.tab_create = 1;
    int horizon = cfg->horizon > 0 ? cfg->horizon : p->horizon;

    Regret R;
    memset(&R, 0, sizeof(R));
    R.vm = &vm;
    R.tab = tab;
    R.horizon = horizon;
    R.explore = explore;
    R.plus = plus;
    rng_seed(R.rng, seed ^ 0xC0FFEEULL);
    /* How many of the learner's decisions are followed in full. Following
     * more closes the gap faster per step (measured on Leduc, liar's dice and
     * tic-tac-toe) for as long as the budget still buys enough rounds: on a
     * five-column four-in-a-row, 800 cheap rounds beat 20 thorough ones. So
     * it starts at none and goes up a level while a round stays cheap. */
    int auto_branch = branch < 0;
    if (auto_branch) branch = 0;
    R.branch = branch;

    Policy pol;
    memset(&pol, 0, sizeof(pol));
    pol.tab = tab;
    int n_checks = curve ? 12 : 0, checks_done = 0;
    if (n_checks) {
        curve_push(curve, 0, eval_mean0(p, &pol, cfg->eval_episodes, horizon));
        checks_done = 1;
    }
    uint64_t ep_seed = seed * 6364136223846793005ULL + 1;
    long iters = 0;
    const double want_rounds = 800.0;
    long since = 0;                 /* rounds played at this level */
    double since_steps = 0.0;
    int came_down = 0;
    while (vm.total_steps < cfg->budget) {
        double round_start = (double)vm.total_steps;
        for (int me = 0; me < p->nagents; me++) {
            if (iters > 0 && vm.total_steps >= cfg->budget) break;
            ep_seed = ep_seed * 6364136223846793005ULL + 1442695040888963407ULL;
            state_reset(p, &s, ep_seed ^ seed);
            s.ep = (int64_t)(ep_seed & 0x7fffffff);
            R.me = me;
            State root = s;
            /* The walk starts before init, so a decision made there learns
             * like any other. Later episodes count for more: the early
             * strategies were poor, and their share of the average should
             * fade. */
            regret_walk(&R, &root, NULL, 0, linear ? (double)(iters + 1) : 1.0, 0, 1);
        }
        iters++;
        if (auto_branch) {
            double cost = (double)vm.total_steps - round_start;
            since++;
            since_steps += cost;
            double mean = since_steps / (double)since, budget = (double)cfg->budget;
            /* Up while this level leaves room, and back down from a level
             * that turns out too dear, which is then not tried again. One
             * round can be unlucky, so alone it must be twice too dear; four
             * are judged on their mean. Trying a level costs little: it is
             * at most `widest` times dearer than one that fitted. */
            if (R.branch > 0 && (cost * want_rounds > 2.0 * budget ||
                                 (since >= 4 && mean * want_rounds > budget))) {
                R.branch--;
                came_down = 1;
                since = 0;
                since_steps = 0.0;
            } else if (!came_down && since >= 2 && R.branch < 64 &&
                       mean * want_rounds <= budget) {
                R.branch++;
                since = 0;
                since_steps = 0.0;
            }
        }
        if (tab->n > max_rows) {
            fprintf(stderr,
                    "hunch: regret keeps one row for everything an agent can be shown, and "
                    "this world has shown %ld different things in %ld decisions.\n"
                    "  It suits worlds whose observations repeat: whole numbers, not "
                    "measurements. Train this one with --algo ppo.\n",
                    tab->n, vm.tab_seen);
            exit(4);
        }
        if (curve && checks_done < n_checks &&
            (double)vm.total_steps / (double)cfg->budget >=
                (double)checks_done / (double)n_checks) {
            curve_push(curve, (double)vm.total_steps,
                       eval_mean0(p, &pol, cfg->eval_episodes, horizon));
            checks_done++;
        }
    }
    if (curve)
        curve_push(curve, (double)vm.total_steps,
                   eval_mean0(p, &pol, cfg->eval_episodes, horizon));
    if (iters_out) *iters_out = iters;
    return (double)vm.total_steps;
}

/* How much the agents could gain between them by each changing alone, with
 * the policy held in the graph. Zero at an equilibrium. */
static double tree_gap(const Tree *T, double *ret, double *best) {
    double r[MAX_AGENTS], gap = 0.0;
    tree_eval(T, 0, r);
    for (int a = 0; a < T->nag; a++) {
        double b = tree_reply(T, a, 0, NULL, NULL);
        if (ret) ret[a] = r[a];
        if (best) best[a] = b;
        gap += b - r[a];
    }
    return gap;
}

/* 1 when each thing any agent can be shown comes up at one node only: nobody
 * is ever unsure where it stands. */
static int tree_nothing_hidden(const Tree *T) {
    long ni = T->isets->n;
    char *met = (char *)xcalloc((size_t)(ni ? ni : 1), 1);
    int ok = 1;
    for (long id = 0; id < T->nn && ok; id++) {
        if (T->n[id].kind != T_DECIDE) continue;
        if (met[T->n[id].iset]) ok = 0;
        met[T->n[id].iset] = 1;
    }
    free(met);
    return ok;
}

/* With nothing hidden there is no need to iterate. Working back from the end,
 * each agent takes the action that pays it most given what everyone does
 * afterwards. No agent can gain by changing alone, anywhere. */
static void tree_lookahead(Tree *T) {
    int nag = T->nag;
    Table *I = T->isets;
    double *val = (double *)xcalloc((size_t)T->nn * (size_t)nag, sizeof(double));
    for (long k = 0; k < T->norder; k++) {
        long id = T->order[k];
        const TNode *n = &T->n[id];
        double *v = val + id * nag;
        if (n->kind == T_CHANCE) {
            for (int j = 0; j < n->nkid; j++) {
                long e = n->kid + j, c = T->kids[e];
                for (int a = 0; a < nag; a++)
                    v[a] += (tree_pays(T, e, a) + val[c * nag + a]) / (double)n->nkid;
            }
        } else if (n->kind == T_DECIDE) {
            int me = n->agent, best = 0;
            double top = -1e300;
            for (int j = 0; j < n->nkid; j++) {
                long e = n->kid + j;
                double x = tree_pays(T, e, me) + val[T->kids[e] * nag + me];
                if (x > top) { top = x; best = j; }
            }
            long e = n->kid + best, c = T->kids[e];
            for (int a = 0; a < nag; a++) v[a] = tree_pays(T, e, a) + val[c * nag + a];
            double *pr = I->ssum + I->aoff[n->iset];
            for (int a = 0; a < I->nact[n->iset]; a++) pr[a] = a == T->kact[e] ? 1.0 : 0.0;
        }
    }
    free(val);
}

/* The same working back, for worlds where an agent is shown the same thing in
 * more than one position: a pile of stones, whatever the step. Each position
 * is valued as if its mover saw the whole state. If, for everything an agent
 * can be shown, one action is among the best in every position where it is
 * shown that, then playing it is as good as seeing everything. With one
 * agent, or two whose rewards sum to zero, that settles the world: neither
 * could do better against an opponent who sees it all, let alone this one.
 * Returns 0, and leaves the policy alone, when some situation has no such
 * action or the world is not of that kind. */
static int tree_lookahead_shared(Tree *T) {
    int nag = T->nag;
    Table *I = T->isets;
    if (nag > 2) return 0;
    if (nag == 2)
        for (long r = 0; r < T->nrews; r += 2)
            if (fabs(T->rews[r] + T->rews[r + 1]) > 1e-12) return 0;
    double *val = (double *)xcalloc((size_t)T->nn * (size_t)nag, sizeof(double));
    char *worse = (char *)xcalloc((size_t)(I->nvals ? I->nvals : 1), 1);
    for (long k = 0; k < T->norder; k++) {
        long id = T->order[k];
        const TNode *n = &T->n[id];
        double *v = val + id * nag;
        if (n->kind == T_CHANCE) {
            for (int j = 0; j < n->nkid; j++) {
                long e = n->kid + j, c = T->kids[e];
                for (int a = 0; a < nag; a++)
                    v[a] += (tree_pays(T, e, a) + val[c * nag + a]) / (double)n->nkid;
            }
        } else if (n->kind == T_DECIDE) {
            int me = n->agent, best = 0;
            double top = -1e300;
            for (int j = 0; j < n->nkid; j++) {
                long e = n->kid + j;
                double x = tree_pays(T, e, me) + val[T->kids[e] * nag + me];
                if (x > top) { top = x; best = j; }
            }
            double slack = 1e-9 * (fabs(top) > 1.0 ? fabs(top) : 1.0);
            for (int j = 0; j < n->nkid; j++) {
                long e = n->kid + j;
                double x = tree_pays(T, e, me) + val[T->kids[e] * nag + me];
                if (x < top - slack) worse[I->aoff[n->iset] + T->kact[e]] = 1;
            }
            long e = n->kid + best, c = T->kids[e];
            for (int a = 0; a < nag; a++) v[a] = tree_pays(T, e, a) + val[c * nag + a];
        }
    }
    free(val);
    int ok = 1;
    for (long is = 0; is < I->n && ok; is++) {
        int any = 0;
        for (int a = 0; a < I->nact[is]; a++)
            if (((I->mask[is] >> a) & 1u) && !worse[I->aoff[is] + a]) any = 1;
        if (!any) ok = 0;
    }
    for (long is = 0; is < I->n && ok; is++) {
        double *pr = I->ssum + I->aoff[is];
        int chosen = -1;
        for (int a = 0; a < I->nact[is]; a++) {
            if (chosen < 0 && ((I->mask[is] >> a) & 1u) && !worse[I->aoff[is] + a]) chosen = a;
            pr[a] = 0.0;
        }
        pr[chosen] = 1.0;
    }
    free(worse);
    return ok;
}

/* One agent alone: replace its policy by the best reply to itself (one action
 * for each thing it can be shown) for as long as that collects more. Returns
 * what the policy it ends with collects. *sure is 1 when that is proven to be
 * the most any policy can. */
static double tree_improve(Tree *T, int *sure) {
    Table *I = T->isets;
    int *pick = (int *)xmalloc(sizeof(int) * (size_t)(I->n ? I->n : 1));
    double ret[MAX_AGENTS];
    tree_eval(T, 0, ret);
    double cur = ret[0];
    int ok = 0;
    for (int round = 0; round < 200; round++) {
        double v = tree_reply(T, 0, 0, &ok, pick);
        if (!(v > cur + 1e-12)) break;
        for (long is = 0; is < I->n; is++) {
            if (pick[is] < 0) continue;
            double *pr = I->ssum + I->aoff[is];
            for (int a = 0; a < I->nact[is]; a++) pr[a] = a == pick[is] ? 1.0 : 0.0;
        }
        cur = v;
    }
    if (!ok && tree_reply(T, 0, 1, NULL, NULL) - cur < 1e-9) ok = 1;
    if (sure) *sure = ok;
    free(pick);
    return cur;
}

/* Regret matching on everything at once. Nothing is sampled: one pass down
 * gives how often each node is reached, one pass up gives the value of every
 * action at every decision, and each situation's regrets are summed over the
 * nodes where it comes up. Regrets are kept from going negative and later
 * iterations weigh more in the average (CFR+, Tammelin 2014). The average
 * strategy is left in the graph, so the graph's own measures apply to it.
 * Returns the iterations run; *gap gets the last measured gap. */
#define TREMBLE 1e-6

static long tree_regret(Tree *T, long max_iters, double target, double max_sec, double *gap,
                        Curve *curve) {
    Table *I = T->isets;
    long nv = I->nvals, nn = T->nn;
    int nag = T->nag;
    double *R = (double *)xcalloc((size_t)(nv ? nv : 1), sizeof(double));
    double *S = (double *)xcalloc((size_t)(nv ? nv : 1), sizeof(double));
    double *sig = (double *)xmalloc(sizeof(double) * (size_t)(nv ? nv : 1));
    double *acc = (double *)xmalloc(sizeof(double) * (size_t)(nv ? nv : 1));
    double *mine = (double *)xmalloc(sizeof(double) * (size_t)nn);
    double *rest = (double *)xmalloc(sizeof(double) * (size_t)nn);
    double *val = (double *)xmalloc(sizeof(double) * (size_t)nn);
    for (long is = 0; is < I->n; is++) probs_uniform(sig + I->aoff[is], I->nact[is], I->mask[is]);
    double t0 = now_sec(), last = 1e300;
    long it = 0, next_check = 10;
    int over = 0;       /* more ways to a node than a double can count */
    while (it < max_iters && !over) {
        it++;
        for (int me = 0; me < nag; me++) {
            /* rest: how often the others and chance bring play to a node.
             * mine: the learner's own odds of coming this way. Both are
             * summed over the ways there, which is what a tree with no shared
             * nodes would give. */
            memset(rest, 0, sizeof(double) * (size_t)nn);
            memset(mine, 0, sizeof(double) * (size_t)nn);
            mine[T->root] = rest[T->root] = 1.0;
            for (long k = T->norder - 1; k >= 0; k--) {
                long id = T->order[k];
                const TNode *n = &T->n[id];
                int own = n->kind == T_DECIDE && n->agent == me;
                const double *pr = n->kind == T_DECIDE ? sig + I->aoff[n->iset] : NULL;
                for (int j = 0; j < n->nkid; j++) {
                    long e = n->kid + j, c = T->kids[e];
                    double q = pr ? pr[T->kact[e]] : 1.0 / (double)n->nkid;
                    /* Every action is counted as played one time in a million
                     * on the way down. A situation that correct play never
                     * reaches is then still weighed, a little, so the policy
                     * learns what to do when someone strays into it instead
                     * of being left at whatever it started as. The values on
                     * the way up are not touched. */
                    if (pr) q = (1.0 - TREMBLE) * q + TREMBLE / (double)n->nkid;
                    mine[c] += own ? mine[id] * q : mine[id];
                    rest[c] += own ? rest[id] : rest[id] * q;
                    if (mine[c] > 1e280) over = 1;
                }
            }
            memset(acc, 0, sizeof(double) * (size_t)(nv ? nv : 1));
            for (long k = 0; k < T->norder; k++) {
                long id = T->order[k];
                const TNode *n = &T->n[id];
                const double *pr = n->kind == T_DECIDE ? sig + I->aoff[n->iset] : NULL;
                double v = 0.0;
                for (int j = 0; j < n->nkid; j++) {
                    long e = n->kid + j;
                    double q = pr ? pr[T->kact[e]] : 1.0 / (double)n->nkid;
                    if (q != 0.0) v += q * (tree_pays(T, e, me) + val[T->kids[e]]);
                }
                val[id] = v;
                if (n->kind != T_DECIDE || n->agent != me) continue;
                long off = I->aoff[n->iset];
                for (int j = 0; j < n->nkid; j++) {
                    long e = n->kid + j;
                    int act = T->kact[e];
                    acc[off + act] += rest[id] * (tree_pays(T, e, me) + val[T->kids[e]] - v);
                    S[off + act] += (double)it * mine[id] * pr[act];
                }
            }
            for (long is = 0; is < I->n; is++) {
                if (I->agent[is] != me) continue;
                long off = I->aoff[is];
                double sum = 0.0;
                for (int a = 0; a < I->nact[is]; a++) {
                    if (!((I->mask[is] >> a) & 1u)) continue;
                    R[off + a] += acc[off + a];
                    if (R[off + a] < 0.0) R[off + a] = 0.0;
                    sum += R[off + a];
                }
                if (sum > 0.0)
                    for (int a = 0; a < I->nact[is]; a++)
                        sig[off + a] = ((I->mask[is] >> a) & 1u) ? R[off + a] / sum : 0.0;
                else
                    probs_uniform(sig + off, I->nact[is], I->mask[is]);
            }
        }
        if (it == next_check || it == max_iters || now_sec() - t0 > max_sec) {
            for (long is = 0; is < I->n; is++) {
                long off = I->aoff[is];
                double sum = 0.0;
                for (int a = 0; a < I->nact[is]; a++) sum += S[off + a];
                if (sum > 0.0)
                    for (int a = 0; a < I->nact[is]; a++) I->ssum[off + a] = S[off + a] / sum;
                else
                    probs_uniform(I->ssum + off, I->nact[is], I->mask[is]);
            }
            double ret[MAX_AGENTS];
            last = tree_gap(T, ret, NULL);
            if (curve && curve->n < curve->cap) {
                curve->pts[curve->n].step = (double)it;
                curve->pts[curve->n].ret = ret[0];
                curve->n++;
            }
            if (g_progress) {
                fprintf(stderr, "gap %ld %.9g\n", it, last);
                fflush(stderr);
            }
            if (last <= target || now_sec() - t0 > max_sec) break;
            next_check = it + (it / 4 > 10 ? it / 4 : 10);
        }
    }
    if (gap) *gap = last;
    free(R); free(S); free(sig); free(acc); free(mine); free(rest); free(val);
    return over ? -1 : it;
}

static void usage(void) {
    fprintf(stderr,
        "usage: hunch-vm <command> <program.hbc> [options]\n"
        "\n"
        "  train <prog> [--algo random|reinforce|ppo|fork|regret|mcts] [--budget N] [--seed S]\n"
        "               [--out policy.hpol] [--eval-episodes N] [--json]\n"
        "               [--think SECONDS | --playouts N]   (mcts: playouts a decision)\n"
        "  report <prog> [--policy p.hpol] [--episodes N] [--exact-nodes N] [--score-sec S]\n"
        "               (what the policy does and collects, as one JSON object)\n"
        "  why   <prog> [--policy p.hpol] [--seed S] [--samples N] [--step K]\n"
        "  trace <prog> [--policy p.hpol] [--seed S] [--forks] [--out file.json]\n"
        "               [--agent alice --actions 1,0]   (interactive play)\n"
        "               [--strict]   (report assert runs, clamped indices and\n"
        "                             zero divisors as one JSON line on stderr)\n"
        "  eval  <prog> [--policy p.hpol] [--episodes N] [--seed S]\n"
        "  speed <prog> [--episodes N]\n");
}

static void apply_common_flags(Cfg *cfg, int argc, char **argv) {
    cfg->budget = arg_long(argc, argv, "--budget", cfg->budget);
    cfg->lr = arg_double(argc, argv, "--lr", cfg->lr);
    cfg->gamma = arg_double(argc, argv, "--gamma", cfg->gamma);
    cfg->lambda = arg_double(argc, argv, "--lambda", cfg->lambda);
    cfg->clip = arg_double(argc, argv, "--clip", cfg->clip);
    cfg->entropy = arg_double(argc, argv, "--entropy", cfg->entropy);
    cfg->fork_ratio = arg_double(argc, argv, "--fork-ratio", cfg->fork_ratio);
    cfg->fork_decay = arg_double(argc, argv, "--fork-decay", cfg->fork_decay);
    cfg->harvest = (int)arg_long(argc, argv, "--harvest", cfg->harvest);
    cfg->fork_pg = (int)arg_long(argc, argv, "--fork-pg", cfg->fork_pg);
    cfg->fork_inject = (int)arg_long(argc, argv, "--fork-inject", cfg->fork_inject);
    cfg->fork_max_points = (int)arg_long(argc, argv, "--fork-points", cfg->fork_max_points);
    cfg->fork_horizon = (int)arg_long(argc, argv, "--fork-horizon", cfg->fork_horizon);
    cfg->ppo_epochs = (int)arg_long(argc, argv, "--epochs", cfg->ppo_epochs);
    cfg->rollout_steps = (int)arg_long(argc, argv, "--rollout", cfg->rollout_steps);
    cfg->eval_episodes = (int)arg_long(argc, argv, "--eval-episodes", cfg->eval_episodes);
    cfg->horizon = (int)arg_long(argc, argv, "--horizon", cfg->horizon);
    cfg->verbose = has_flag(argc, argv, "--quiet") ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/* Library: a world stepped from outside                               */
/* ------------------------------------------------------------------ */

/* Built with -DHUNCH_LIB -shared, this file is also what `import hunch`
 * steps environments with. Some agents are decided from outside (a learner,
 * a person, a language model); the rest play a policy file or at random.
 *
 * A block (init, or one step) that reaches an outside decision stops there.
 * When the action arrives the block is run again from its start with the
 * outside actions so far named in advance: the state holds the random
 * stream, so the replay repeats everything else exactly. Nothing about the
 * interpreter has to be suspended. */

#define ENV_PLAN 64     /* decisions in one block */

typedef struct {
    State s, start;             /* the working state; the start of the current block */
    /* Every decision already made in this block, by anyone, in order. A slot
     * is therefore the same instant in any environment of its world, whoever
     * that environment lets decide. */
    int plan[ENV_PLAN];
    int nplan;
    /* Bit k of reseed_at: right after decision k of this block, the random stream
     * starts again from that seed. */
    uint64_t reseed[ENV_PLAN];
    uint64_t reseed_at;
    /* A hash of where in its block the waiting decision is asked: the
     * instruction, the stack, the calls it is inside, what the block has paid. */
    uint64_t ev_key;
    int in_init, waiting, over, truncated;
    int ev_agent, ev_n, ev_obs_len;
    uint32_t ev_legal;
    double ev_obs[MAX_OBS];
    double reward[MAX_AGENTS];  /* paid and not yet collected */
    uint64_t episode;           /* vec: episodes started in this slot */
} Slot;

typedef struct {
    Driver dr;                  /* first, so a Driver * is an EnvDriver * */
    uint32_t outside;
    Slot *slot;
    int seen;                   /* decisions made so far in this run of the block */
    int made[ENV_PLAN];         /* and what each was */
    int free_run;               /* nobody is asked: play on to the end */
    int drawn_fresh;            /* this decision was drawn from a stream just started */
} EnvDriver;

typedef struct {
    Program *p;                 /* the world's, not owned */
    Policy pol;
    VM vm;
    EnvDriver dr;
    Slot *slots;
    Slot undo;                  /* the slot being stepped, as it was before the step */
    int n, horizon;
    uint64_t vec_seed;
    char err[320];
} HunchEnv;

/* Only the slots the program uses are ever nonzero. */
static void state_copy(const Program *p, State *dst, const State *src) {
    memcpy(dst, src, offsetof(State, v) + sizeof(double) * (size_t)p->nslots);
}

static uint64_t hash_mix(uint64_t h, const void *data, size_t len) {
    const unsigned char *b = (const unsigned char *)data;
    for (size_t i = 0; i < len; i++) { h ^= b[i]; h *= 0x100000001B3ULL; }
    return h ^ (h >> 29);
}

/* A slot, copying only the part of each State the world uses. */
static void slot_copy(const Program *p, Slot *dst, const Slot *src) {
    state_copy(p, &dst->s, &src->s);
    state_copy(p, &dst->start, &src->start);
    memcpy((char *)dst + offsetof(Slot, plan), (const char *)src + offsetof(Slot, plan),
           sizeof(Slot) - offsetof(Slot, plan));
}

static int env_decide(Driver *dr, VM *vm, int agent, const double *obs, int obs_len,
                      int n_act, uint32_t legal, const double *probs) {
    EnvDriver *e = (EnvDriver *)dr;
    (void)probs;
    Slot *sl = e->slot;
    if (e->seen < sl->nplan) return sl->plan[e->seen];
    if (e->free_run) {
        /* The decision the slot waited at is drawn here, so a new stream
         * asked for at it has to start before the draw, not after. */
        if (e->seen == sl->nplan && e->seen < ENV_PLAN && ((sl->reseed_at >> e->seen) & 1u)) {
            rng_seed(vm->s->rng, sl->reseed[e->seen]);
            e->drawn_fresh = 1;
        }
        return DRIVE_PASS;
    }
    if (!((e->outside >> agent) & 1u)) return DRIVE_PASS;
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ (uint64_t)vm->at_ip;
    h = hash_mix(h, &vm->sp, sizeof(vm->sp));
    h = hash_mix(h, vm->stack, sizeof(double) * (size_t)vm->sp);
    h = hash_mix(h, vm->call_ip, sizeof(int) * (size_t)vm->csp);
    h = hash_mix(h, vm->agent_reward, sizeof(double) * (size_t)vm->p->nagents);
    sl->ev_key = h;
    sl->ev_agent = agent;
    sl->ev_n = n_act;
    sl->ev_legal = legal;
    sl->ev_obs_len = obs_len;
    memcpy(sl->ev_obs, obs, sizeof(double) * (size_t)obs_len);
    return DRIVE_STOP;
}

static void env_decided(Driver *dr, VM *vm, int agent, int n_act, uint32_t legal,
                        const double *probs, int action) {
    EnvDriver *e = (EnvDriver *)dr;
    (void)agent; (void)n_act; (void)legal; (void)probs;
    int k = e->seen++;
    if (k >= ENV_PLAN) fatal(2, "more than %d decisions in one step", ENV_PLAN);
    e->made[k] = action;
    if (e->drawn_fresh) { e->drawn_fresh = 0; return; }
    if ((e->slot->reseed_at >> k) & 1u) rng_seed(vm->s->rng, e->slot->reseed[k]);
}

/* Run on until an outside agent must decide or the episode is over. */
static void env_advance(HunchEnv *e, Slot *sl) {
    VM *vm = &e->vm;
    const Program *p = e->p;
    sl->waiting = 0;
    for (;;) {
        state_copy(p, &sl->s, &sl->start);
        vm->s = &sl->s;
        e->dr.slot = sl;
        e->dr.seen = 0;
        vm->driver = &e->dr.dr;
        if (sl->in_init) {
            vm_run(vm, p->init_entry);
            /* The runtime drops what init pays, here as everywhere. */
            memset(vm->agent_reward, 0, sizeof(vm->agent_reward));
        } else {
            vm_step(vm);
        }
        vm->driver = NULL;
        int stopped = vm->stopped;
        vm->stopped = 0;
        if (stopped) {
            /* What the others did on the way here is now part of the plan. */
            for (int k = sl->nplan; k < e->dr.seen; k++) sl->plan[k] = e->dr.made[k];
            sl->nplan = e->dr.seen;
            sl->waiting = 1;
            return;
        }
        for (int a = 0; a < p->nagents; a++) sl->reward[a] += vm->agent_reward[a];
        sl->in_init = 0;
        sl->nplan = 0;
        sl->reseed_at = 0;
        state_copy(p, &sl->start, &sl->s);
        if (sl->s.done) { sl->over = 1; return; }
        if (sl->s.step >= e->horizon) { sl->over = 1; sl->truncated = 1; return; }
    }
}

static void env_begin(HunchEnv *e, Slot *sl, uint64_t seed) {
    state_reset(e->p, &sl->start, seed);
    sl->start.ep = 0;
    sl->nplan = 0;
    sl->reseed_at = 0;
    sl->in_init = 1;
    sl->over = sl->truncated = 0;
    memset(sl->reward, 0, sizeof(sl->reward));
    env_advance(e, sl);
}

static void trap_copy(char *err, int errlen) {
    if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s", g_trap_msg);
}

/* Every entry point runs under a trap: a failed assert or an illegal action
 * comes back as an error code and a message, not as the end of the process. */
#define TRAPPED(on_fail)                        \
    g_trap_on = 1;                              \
    if (setjmp(g_trap)) { g_trap_on = 0; on_fail; }

void *hunch_world_open(const char *hbc, char *err, int errlen) {
    Program *p = (Program *)calloc(1, sizeof(Program));
    TRAPPED({ trap_copy(err, errlen); free(p); return NULL; })
    load_program(hbc, p);
    g_trap_on = 0;
    return p;
}

void hunch_world_close(void *world) {
    Program *p = (Program *)world;
    if (!p) return;
    free(p->consts); free(p->code); free(p->defaults);
    free(p);
}

/* n environments on one world. `outside`: bit a set = agent a is decided by
 * the caller. `policy`: a .hpol for the others, or NULL for uniform play. */
void *hunch_env_new(void *world, int n, uint32_t outside, const char *policy, int horizon,
                    char *err, int errlen) {
    Program *p = (Program *)world;
    HunchEnv *e = (HunchEnv *)calloc(1, sizeof(HunchEnv));
    TRAPPED({ trap_copy(err, errlen); policy_free(&e->pol); free(e->slots); free(e);
              return NULL; })
    if (n < 1) fatal(2, "an environment needs at least one slot, got %d", n);
    e->p = p;
    e->n = n;
    e->horizon = horizon > 0 ? horizon : p->horizon;
    e->slots = (Slot *)xcalloc((size_t)e->n, sizeof(Slot));
    vm_init(&e->vm, p, &e->slots[0].s);
    if (policy && policy[0]) {
        if (!policy_load(policy, p, &e->pol)) {
            snprintf(g_trap_msg, sizeof(g_trap_msg),
                     "could not read policy %s: it does not fit this world", policy);
            longjmp(g_trap, 1);
        }
        policy_attach(&e->vm, &e->pol);
        if (e->pol.search) e->pol.search->horizon = e->horizon;
    }
    e->dr.dr.decide = env_decide;
    e->dr.dr.decided = env_decided;
    e->dr.outside = outside;
    g_trap_on = 0;
    return e;
}

void hunch_env_free(void *env) {
    HunchEnv *e = (HunchEnv *)env;
    if (!e) return;
    policy_free(&e->pol);
    free(e->slots);
    free(e);
}

const char *hunch_env_error(void *env) { return ((HunchEnv *)env)->err; }

/* Agents whose bit is set play uniformly, whatever the policy says. */
void hunch_env_set_random(void *env, uint32_t agents) {
    ((HunchEnv *)env)->vm.random_agents = agents;
}

#define ENV_TRAPPED(e)  TRAPPED({ trap_copy((e)->err, (int)sizeof((e)->err)); return -1; })

/* Start the episode `seed` names. 0, or -1 with the reason in hunch_env_error. */
int hunch_env_reset(void *env, int i, uint64_t seed) {
    HunchEnv *e = (HunchEnv *)env;
    ENV_TRAPPED(e)
    env_begin(e, &e->slots[i], seed);
    g_trap_on = 0;
    return 0;
}

/* Why `action` cannot be taken in slot sl now, as a fatal error; nothing has
 * changed when it is raised. */
static void env_check(const HunchEnv *e, const Slot *sl, int i, int action) {
    char where[32] = "";
    if (e->n > 1) snprintf(where, sizeof(where), "slot %d: ", i);
    if (!sl->waiting)
        fatal(2, "%sno decision is waiting: the episode is over or was never reset", where);
    if (action < 0 || action >= sl->ev_n || !((sl->ev_legal >> action) & 1u))
        fatal(3, "%saction %d is not legal here for '%s'", where, action,
              e->p->agents[sl->ev_agent].name);
    if (sl->nplan >= ENV_PLAN)
        fatal(2, "%smore than %d decisions in one step", where, ENV_PLAN);
}

/* Take `action` in slot i. If the world fails on the way to the next decision
 * (an assert, say), the slot is put back as it was: still waiting here. */
int hunch_env_act(void *env, int i, int action) {
    HunchEnv *e = (HunchEnv *)env;
    Slot *sl = &e->slots[i];
    volatile int stepping = 0;
    TRAPPED({ trap_copy(e->err, (int)sizeof(e->err));
              if (stepping) slot_copy(e->p, &e->slots[i], &e->undo);
              return -1; })
    env_check(e, sl, i, action);
    slot_copy(e->p, &e->undo, sl);
    stepping = 1;
    sl->plan[sl->nplan++] = action;
    env_advance(e, sl);
    g_trap_on = 0;
    return 0;
}

/* The agent that must decide, or -1 when the episode is over. */
int hunch_env_agent(void *env, int i) {
    Slot *sl = &((HunchEnv *)env)->slots[i];
    return sl->waiting ? sl->ev_agent : -1;
}

uint32_t hunch_env_legal(void *env, int i) {
    Slot *sl = &((HunchEnv *)env)->slots[i];
    return sl->waiting ? sl->ev_legal : 0;
}

/* What the deciding agent is shown. Returns how many values. */
int hunch_env_obs(void *env, int i, double *out) {
    Slot *sl = &((HunchEnv *)env)->slots[i];
    if (!sl->waiting) return 0;
    memcpy(out, sl->ev_obs, sizeof(double) * (size_t)sl->ev_obs_len);
    return sl->ev_obs_len;
}

/* 0 while it runs, 1 once it has ended by done(), 2 at the horizon. */
int hunch_env_over(void *env, int i) {
    Slot *sl = &((HunchEnv *)env)->slots[i];
    return !sl->over ? 0 : sl->truncated ? 2 : 1;
}

/* What each agent has been paid since this was last called with take. */
void hunch_env_rewards(void *env, int i, double *out, int take) {
    HunchEnv *e = (HunchEnv *)env;
    Slot *sl = &e->slots[i];
    for (int a = 0; a < e->p->nagents; a++) out[a] = sl->reward[a];
    if (take) memset(sl->reward, 0, sizeof(sl->reward));
}

/* Every slot of the world's memory, as it stood when the current step began:
 * the declared state first. Returns how many values. */
int hunch_env_state(void *env, int i, double *out) {
    HunchEnv *e = (HunchEnv *)env;
    memcpy(out, e->slots[i].start.v, sizeof(double) * (size_t)e->p->nslots);
    return e->p->nslots;
}

long hunch_env_steps(void *env, int i) { return (long)((HunchEnv *)env)->slots[i].start.step; }

/* A whole instant, random stream included: load it back, into this slot or
 * another, and the future is the same again. That is a fork. */
int hunch_env_snapshot_size(void) { return (int)sizeof(Slot); }
void hunch_env_save(void *env, int i, void *buf) {
    memcpy(buf, &((HunchEnv *)env)->slots[i], sizeof(Slot));
}
void hunch_env_load(void *env, int i, const void *buf) {
    memcpy(&((HunchEnv *)env)->slots[i], buf, sizeof(Slot));
}
/* From the decision the slot waits at, the random numbers come from a new
 * stream: everything up to now stays as it was, what follows is drawn again. */
void hunch_env_reseed(void *env, int i, uint64_t seed) {
    Slot *sl = &((HunchEnv *)env)->slots[i];
    if (!sl->waiting || sl->nplan >= ENV_PLAN) return;
    sl->reseed[sl->nplan] = seed;
    sl->reseed_at |= (uint64_t)1 << sl->nplan;
}

/* With hunch_env_memory and the step number, this names the instant the slot
 * waits at: two slots that agree on all three have the same future. */
uint64_t hunch_env_key(void *env, int i) {
    Slot *sl = &((HunchEnv *)env)->slots[i];
    uint64_t h = sl->waiting ? sl->ev_key : 0;
    int where[4] = { sl->waiting, sl->in_init, sl->waiting ? sl->ev_agent : -1, sl->over };
    return hash_mix(h, where, sizeof(where));
}

/* The world's whole memory at the decision the slot waits at, locals and
 * all. Returns how many values. */
int hunch_env_memory(void *env, int i, double *out) {
    HunchEnv *e = (HunchEnv *)env;
    memcpy(out, e->slots[i].s.v, sizeof(double) * (size_t)e->p->nslots);
    return e->p->nslots;
}

/* Play slot i to the end of its episode with every agent choosing uniformly
 * among its legal actions, nobody asked. out: what each agent is paid from
 * here on, with anything not yet collected. */
int hunch_env_playout(void *env, int i, double *out) {
    HunchEnv *e = (HunchEnv *)env;
    Slot *sl = &e->slots[i];
    uint32_t was = e->vm.random_agents;
    TRAPPED({ trap_copy(e->err, (int)sizeof(e->err)); e->dr.free_run = 0;
              e->vm.random_agents = was; return -1; })
    if (sl->waiting) {
        e->dr.free_run = 1;
        e->vm.random_agents = mask_all(MAX_AGENTS);
        env_advance(e, sl);
        e->dr.free_run = 0;
        e->vm.random_agents = was;
    }
    for (int a = 0; a < e->p->nagents; a++) out[a] = sl->reward[a];
    memset(sl->reward, 0, sizeof(sl->reward));
    g_trap_on = 0;
    return 0;
}

/* What the environment's policy does where `agent` is shown `obs`: 1, or 0
 * for a situation it has no answer for (uniform). -1: no legal action. -2:
 * the policy is a search, which chooses from an instant of the world, not
 * from what an agent is shown, and so has no answer to this question. */
int hunch_env_policy_probs(void *env, int agent, const double *obs, uint32_t legal,
                           double *out) {
    HunchEnv *e = (HunchEnv *)env;
    const Program *p = e->p;
    if (agent < 0 || agent >= p->nagents) return -1;
    int n = p->agents[agent].n_act, dim = p->agents[agent].obs_dim;
    legal &= mask_all(n);
    if (!legal) return -1;
    if (e->pol.search) {
        snprintf(e->err, sizeof(e->err),
                 "this policy is a search: it chooses by forking the world at the instant of "
                 "a decision, not from what the agent is shown, so it has no probabilities "
                 "for an observation. Play it in an environment to see what it does");
        return -2;
    }
    if (e->pol.tab) {
        long row = tab_find(e->pol.tab, agent, obs, dim, n, legal, 0);
        if (row < 0) { probs_uniform(out, n, legal); return 0; }
        tab_average(e->pol.tab, row, out);
        return 1;
    }
    if (e->pol.nets[agent]) {
        Net *net = e->pol.nets[agent];
        net_forward(net, obs);
        net_mask(net, legal);
        memcpy(out, net->probs, sizeof(double) * (size_t)n);
        return 1;
    }
    probs_uniform(out, n, legal);
    return 0;
}

/* ---- all slots at once, for learners that want arrays ---- */

static uint64_t vec_seed(const HunchEnv *e, int i, uint64_t episode) {
    uint64_t z = e->vec_seed + 0x9E3779B97F4A7C15ULL * ((uint64_t)i + 1) +
                 0xD1B54A32D192ED03ULL * episode;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Start a new episode in slot i, skipping any that end before an outside
 * agent decides (what they paid is dropped). */
static void vec_begin(HunchEnv *e, int i) {
    Slot *sl = &e->slots[i];
    for (int tries = 0; tries < 100000; tries++) {
        env_begin(e, sl, vec_seed(e, i, sl->episode++));
        if (sl->waiting) return;
    }
    fatal(2, "no episode of this world reaches a decision of an outside agent");
}

static void vec_show(HunchEnv *e, int i, double *obs, int width, uint32_t *legal,
                     int32_t *agent) {
    Slot *sl = &e->slots[i];
    if (obs) {
        double *row = obs + (size_t)i * (size_t)width;
        int k = sl->ev_obs_len < width ? sl->ev_obs_len : width;
        memcpy(row, sl->ev_obs, sizeof(double) * (size_t)k);
        for (; k < width; k++) row[k] = 0.0;
    }
    if (legal) legal[i] = sl->ev_legal;
    if (agent) agent[i] = sl->ev_agent;
}

/* obs: n rows of `width` values. legal, agent: n values each; either may be NULL. */
int hunch_vec_reset(void *env, uint64_t seed, double *obs, int width, uint32_t *legal,
                    int32_t *agent) {
    HunchEnv *e = (HunchEnv *)env;
    ENV_TRAPPED(e)
    e->vec_seed = seed;
    for (int i = 0; i < e->n; i++) {
        e->slots[i].episode = 0;
        vec_begin(e, i);
        vec_show(e, i, obs, width, legal, agent);
    }
    g_trap_on = 0;
    return 0;
}

/* One action for each slot. rew: n rows of one value per agent of the world,
 * what each was paid since the slot's last step. over: 0, 1 (done) or 2
 * (horizon); a slot whose episode ended has already started the next, and
 * obs, legal and agent describe that new episode's first decision. */
int hunch_vec_step(void *env, const int32_t *actions, double *obs, int width, double *rew,
                   uint8_t *over, uint32_t *legal, int32_t *agent) {
    HunchEnv *e = (HunchEnv *)env;
    int nag = e->p->nagents;
    volatile int stepping = 0;
    /* If the world itself fails in a row (an assert, say), some rows have
     * stepped and some have not. Keeping a copy of every row to undo that
     * would cost a third of the speed, so the batch is stopped instead: no
     * row is waiting until hunch_vec_reset. */
    TRAPPED({ trap_copy(e->err, (int)sizeof(e->err));
              if (stepping)
                  for (int k = 0; k < e->n; k++) e->slots[k].waiting = 0;
              return -1; })
    /* Every row is checked before any is stepped, so a refused action leaves
     * them all where they were. */
    for (int i = 0; i < e->n; i++) env_check(e, &e->slots[i], i, actions[i]);
    stepping = 1;
    for (int i = 0; i < e->n; i++) {
        Slot *sl = &e->slots[i];
        sl->plan[sl->nplan++] = actions[i];
        env_advance(e, sl);
        for (int a = 0; a < nag; a++) rew[(size_t)i * (size_t)nag + a] = sl->reward[a];
        memset(sl->reward, 0, sizeof(sl->reward));
        over[i] = (uint8_t)(!sl->over ? 0 : sl->truncated ? 2 : 1);
        if (sl->over) vec_begin(e, i);
        vec_show(e, i, obs, width, legal, agent);
    }
    g_trap_on = 0;
    return 0;
}

/* Uniform random play in every slot for `steps` calls of step, all inside
 * the library: how fast the world itself runs. Returns decisions made. */
long hunch_vec_random(void *env, long steps, uint64_t seed) {
    HunchEnv *e = (HunchEnv *)env;
    uint64_t rng[4];
    rng_seed(rng, seed);
    long made = 0;
    ENV_TRAPPED(e)
    for (int i = 0; i < e->n; i++)
        if (!e->slots[i].waiting) vec_begin(e, i);
    for (long t = 0; t < steps; t++) {
        for (int i = 0; i < e->n; i++) {
            Slot *sl = &e->slots[i];
            int k = (int)(rng_double(rng) * mask_count(sl->ev_legal, sl->ev_n)), action = 0;
            for (int a = 0; a < sl->ev_n; a++) {
                if (!((sl->ev_legal >> a) & 1u)) continue;
                action = a;
                if (k-- == 0) break;
            }
            env_check(e, sl, i, action);
            sl->plan[sl->nplan++] = action;
            env_advance(e, sl);
            memset(sl->reward, 0, sizeof(sl->reward));
            if (sl->over) vec_begin(e, i);
            made++;
        }
    }
    g_trap_on = 0;
    return made;
}

#ifndef HUNCH_LIB
int main(int argc, char **argv) {
    if (argc < 3) { usage(); return 1; }
    /* Following every outcome recurses once per event of an episode. */
    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0) {
        rlim_t want = (rlim_t)1 << 30;
        if (rl.rlim_max != RLIM_INFINITY && rl.rlim_max < want) want = rl.rlim_max;
        if (rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < want) {
            rl.rlim_cur = want;
            setrlimit(RLIMIT_STACK, &rl);
        }
    }
    const char *cmd = argv[1];

    Program p;
    load_program(argv[2], &p);

    Cfg cfg;
    cfg_defaults(&cfg);
    apply_common_flags(&cfg, argc, argv);

    uint64_t seed = (uint64_t)arg_long(argc, argv, "--seed", 1);
    g_progress = has_flag(argc, argv, "--progress");
    const char *algo = arg_value(argc, argv, "--algo", "ppo");
    const char *outpath = arg_value(argc, argv, "--out", NULL);
    const char *polpath = arg_value(argc, argv, "--policy", NULL);
    int horizon = cfg.horizon > 0 ? cfg.horizon : p.horizon;

    if (strcmp(cmd, "train") == 0 || strcmp(cmd, "bench") == 0) {
        Net *nets[MAX_AGENTS];
        for (int i = 0; i < MAX_AGENTS; i++) nets[i] = NULL;
        Curve curve;
        curve.cap = 64;
        curve.n = 0;
        curve.pts = (CurvePoint *)xcalloc(64, sizeof(CurvePoint));

        double t0 = now_sec();
        double steps, final_ret;
        long iters = 0, rows = 0;
        int whole = 0;
        double gap = 0.0;
        long positions = 0;
        char not_whole[64] = "";
        int proven = 0;
        int hidden = -1;            /* HIDDEN_*, once the world has been checked */
        long checked[4] = {0, 0, 0, 0};
        int playouts = 0;
        double playouts_sec = 0.0, think = 0.0, explore = 0.0;
        Learn learn;
        memset(&learn, 0, sizeof(learn));
        Guide *guide = NULL;
        /* auto: the exact answer when the world is small enough to follow
         * whole; failing that, a search at play time when sampling finds
         * nothing hidden from the agents; failing that, a network for one
         * agent and sampled regret matching for several. */
        int is_auto = strcmp(algo, "auto") == 0;
        if (is_auto) algo = "regret";
        if (strcmp(algo, "regret") == 0) {
            /* Small enough to hold whole: no sampling, and it stops when no
             * agent can gain more than the target by changing alone. */
            long max_nodes = arg_long(argc, argv, "--exact-nodes", 2000000);
            Tree *T = max_nodes > 0
                ? tree_build(&p, NULL, horizon, max_nodes, !has_flag(argc, argv, "--no-merge"))
                : NULL;
            if (T && !T->fail) {
                positions = T->nn;
                int look = !has_flag(argc, argv, "--no-lookahead");
                if (look && tree_nothing_hidden(T)) {
                    whole = 2;
                    tree_lookahead(T);
                    gap = tree_gap(T, NULL, NULL);
                } else if (look && tree_lookahead_shared(T)) {
                    whole = 2;
                    gap = tree_gap(T, NULL, NULL);
                } else if (p.nagents == 1 && !has_flag(argc, argv, "--no-lookahead")) {
                    /* Best replies from random play; when that is not proven
                     * best, also from where regret matching ends, and the
                     * better of the two is kept. */
                    whole = 3;
                    Table *I = T->isets;
                    size_t bytes = sizeof(double) * (size_t)(I->nvals ? I->nvals : 1);
                    int sure = 0;
                    double v1 = tree_improve(T, &sure);
                    if (!sure) {
                        double *keep = (double *)xmalloc(bytes);
                        memcpy(keep, I->ssum, bytes);
                        long n = tree_regret(T, arg_long(argc, argv, "--iters", 100000),
                                             arg_double(argc, argv, "--target", 0.001),
                                             arg_double(argc, argv, "--max-sec", 5.0), &gap,
                                             NULL);
                        iters = n > 0 ? n : 0;
                        if (n < 0 || tree_improve(T, &sure) < v1) {
                            memcpy(I->ssum, keep, bytes);
                            sure = 0;
                        }
                        free(keep);
                    }
                    proven = sure;
                    gap = tree_gap(T, NULL, NULL);
                } else {
                    iters = tree_regret(T, arg_long(argc, argv, "--iters", 100000),
                                        arg_double(argc, argv, "--target", 0.001),
                                        arg_double(argc, argv, "--max-sec", 60.0), &gap, &curve);
                    whole = iters >= 0;
                }
            }
            if (T && T->fail == TREE_RAND)
                snprintf(not_whole, sizeof(not_whole), "rand:%d:%d", T->rand_line, T->rand_col);
            else if (!T) snprintf(not_whole, sizeof(not_whole), "off");
            else if (!whole) snprintf(not_whole, sizeof(not_whole), "big:%ld", max_nodes);
            if (!whole && is_auto) {
                /* A search reads the whole state, so it is chosen only on
                 * evidence that the agents are shown all of it. */
                hidden = shown_check(&p, horizon, seed, checked);
                if (hidden == HIDDEN_NOTHING) algo = "mcts";
            }
            if (!whole && is_auto && p.nagents == 1 && hidden != HIDDEN_NOTHING) algo = "ppo";
            if (!whole && strcmp(algo, "regret") == 0 && !arg_value(argc, argv, "--budget", NULL))
                cfg.budget = 2000000;
            if (whole) {
                rows = T->isets->n;
                if (outpath) save_table(outpath, &p, T->isets);
                double ret[MAX_AGENTS];
                tree_eval(T, 0, ret);
                final_ret = ret[0];
                steps = 0;
            }
            tree_free(T);
        }
        if (whole) {
            /* done above */
        } else if (strcmp(algo, "mcts") == 0) {
            /* Nothing is learned ahead of play. What is settled here is how
             * many playouts a decision gets: this world's playouts a second
             * are measured on this machine, and the count that fills --think
             * seconds (a tenth by default) is written to the file, so the
             * policy plays the same on any machine. --playouts sets it. */
            if (hidden < 0) hidden = shown_check(&p, horizon, seed, checked);
            /* The network is made first (unless --games 0), by self-play, and
             * the playouts a decision are measured with it. */
            learn.games = arg_long(argc, argv, "--games", 0);
            /* a count of games alone has no time limit */
            learn.seconds = arg_double(argc, argv, "--train-sec",
                                       learn.games > 0 ? HUGE_VAL : GUIDE_TRAIN_SEC);
            learn.cores = (int)arg_long(argc, argv, "--cores", 0);
            if (learn.cores < 1) learn.cores = cores_online();
            /* Self-play leaves the random playouts of a leaf out: it plays
             * more games that way, and learned no better with them. */
            learn.round = 64;
            learn.playouts = 200;
            learn.hid = 64;
            learn.sample_moves = 6;
            learn.batch = 64;
            learn.window = 100000;
            learn.reuse = 4.0;
            learn.lr = 1e-3;
            learn.noise = 0.25;
            int learning = !(learn.games == 0 && arg_value(argc, argv, "--games", NULL)) &&
                           learn.seconds > 0;
            explore = arg_double(argc, argv, "--explore", learning ? GUIDE_EXPLORE : 1.0);
            if (!(explore >= 0.0)) explore = learning ? GUIDE_EXPLORE : 1.0;
            learn.explore = explore;
            if (learning) guide = guide_train(&p, horizon, seed, hidden, &learn);
            think = arg_double(argc, argv, "--think", 0.1);
            if (!(think > 0.0)) think = 0.1;
            playouts_sec = search_measure(&p, horizon, explore, seed, 0.02, 48, 1.0, guide,
                                          GUIDE_ROLLOUT);
            playouts = (int)arg_long(argc, argv, "--playouts", 0);
            if (playouts < 1) {
                double want = playouts_sec > 0 ? playouts_sec * think : 1000.0;
                /* two figures are all the measurement is good for */
                double unit = pow(10.0, floor(log10(want > 10.0 ? want : 10.0)) - 1.0);
                want = floor(want / unit + 0.5) * unit;
                playouts = want < 16.0 ? 16 : want > 5e7 ? 50000000 : (int)want;
            }
            Search *S = search_new(&p, playouts, explore, hidden);
            memcpy(S->checked, checked, sizeof(checked));
            S->guide = guide;
            if (guide) S->mix = GUIDE_ROLLOUT;
            if (outpath) save_search(outpath, &p, S);
            S->guide = NULL;        /* still wanted for --json below */
            search_free(S);
            steps = 0;
            final_ret = 0.0;        /* not measured: see the report */
        } else if (strcmp(algo, "regret") == 0) {
            Policy pol;
            memset(&pol, 0, sizeof(pol));
            pol.tab = tab_new();
            steps = train_regret(&p, &cfg, seed, pol.tab, &curve, &iters,
                                 (int)arg_long(argc, argv, "--branch", -1),
                                 arg_double(argc, argv, "--explore", 0.6),
                                 arg_long(argc, argv, "--max-rows", 2000000),
                                 (int)arg_long(argc, argv, "--linear", 1),
                                 (int)arg_long(argc, argv, "--plus", 0));
            rows = pol.tab->n;
            if (outpath) save_table(outpath, &p, pol.tab);
            final_ret = eval_mean0(&p, &pol, cfg.eval_episodes, horizon);
            tab_free(pol.tab);
        } else {
            steps = train(&p, algo, &cfg, seed, nets, &curve);
            if (outpath && strcmp(algo, "random") != 0) save_policy(outpath, &p, nets);
            final_ret = eval_policy(&p, nets, cfg.eval_episodes, EVAL_SEED, horizon, 0);
        }
        double wall = now_sec() - t0;

        if (has_flag(argc, argv, "--json")) {
            printf("{\"algo\":\"%s\",\"seed\":%llu,\"budget\":%ld,", algo,
                   (unsigned long long)seed, cfg.budget);
            printf("\"steps_used\":%.0f,\"wall_sec\":%.3f,", steps, wall);
            /* A search is scored by the report, not here. */
            if (strcmp(algo, "mcts") == 0) printf("\"final_return\":null");
            else printf("\"final_return\":%.9g", final_ret);
            if (not_whole[0]) printf(",\"not_whole\":\"%s\"", not_whole);
            if (hidden >= 0)
                printf(",\"hidden\":\"%s\",\"hidden_check\":{\"positions\":%ld,\"repeats\":%ld,"
                       "\"distinct\":%ld,\"conflicts\":%ld,\"min_repeats\":%ld,"
                       "\"min_share\":%.9g,\"min_distinct\":%ld}", HIDDEN_NAME[hidden],
                       checked[0], checked[1], checked[2], checked[3], SHOWN_MIN_REPEATS,
                       SHOWN_MIN_SHARE, SHOWN_MIN_DISTINCT);
            if (strcmp(algo, "mcts") == 0)
                printf(",\"method\":\"mcts\",\"chosen\":%s,\"playouts\":%d,"
                       "\"playouts_per_sec\":%.0f,\"think\":%.9g,\"playouts_set\":%s,"
                       "\"explore\":%.9g", is_auto ? "true" : "false", playouts, playouts_sec,
                       think, arg_long(argc, argv, "--playouts", 0) >= 1 ? "true" : "false",
                       explore);
            if (strcmp(algo, "mcts") == 0) {
                printf(",\"learned\":");
                json_guide(stdout, guide);
                printf(",\"rollout\":%.9g", guide ? GUIDE_ROLLOUT : 1.0);
            }
            if (whole == 3) printf(",\"proven\":%s", proven ? "true" : "false");
            if (strcmp(algo, "regret") == 0)
                printf(",\"method\":\"%s\",\"iterations\":%ld,\"situations\":%ld,"
                       "\"positions\":%ld,\"gap\":%.9g,\"target\":%.9g",
                       whole == 2 ? "lookahead" : whole == 3 ? "search" : whole ? "regret"
                           : "sampled", iters, rows, positions, gap,
                       arg_double(argc, argv, "--target", 0.001));
            printf(",\"curve\":[");
            for (int i = 0; i < curve.n; i++) {
                if (i) printf(",");
                printf("{\"step\":%.0f,\"return\":%.9g}", curve.pts[i].step,
                       curve.pts[i].ret);
            }
            printf("]}\n");
        } else if (cfg.verbose) {
            fprintf(stderr, "algo=%-9s seed=%llu steps=%.0f wall=%.2fs return=%.4f\n",
                    algo, (unsigned long long)seed, steps, wall, final_ret);
            for (int i = 0; i < curve.n; i++)
                fprintf(stderr, "  %8.0f  %10.4f\n", curve.pts[i].step, curve.pts[i].ret);
        }
        for (int i = 0; i < MAX_AGENTS; i++) net_free(nets[i]);
        free(curve.pts);
        guide_free(guide);
        return 0;
    }

    if (strcmp(cmd, "report") == 0 || strcmp(cmd, "why") == 0) {
        Policy pol;
        memset(&pol, 0, sizeof(pol));
        if (polpath && !policy_load(polpath, &p, &pol)) {
            fprintf(stderr, "hunch-vm: could not read policy %s\n", polpath);
            return 1;
        }
        if (pol.search) pol.search->horizon = horizon;
        int rc = strcmp(cmd, "why") == 0
            ? cmd_why(&p, &pol, horizon, seed, argc, argv)
            : cmd_report(&p, &pol, horizon, seed, argc, argv);
        policy_free(&pol);
        return rc;
    }

    if (strcmp(cmd, "trace") == 0) {
        Policy pol;
        memset(&pol, 0, sizeof(pol));
        Net **nets = pol.nets;
        if (polpath && !policy_load(polpath, &p, &pol)) {
            fprintf(stderr, "hunch-vm: could not read policy %s\n", polpath);
            return 1;
        }

        State s;
        VM vm;
        vm_init(&vm, &p, &s);

        int sagent = -1;
        int *sacts = NULL;
        int sn = 0;
        const char *script_name = arg_value(argc, argv, "--agent", NULL);
        const char *script_actions = arg_value(argc, argv, "--actions", NULL);
        if (script_name) {
            for (int i = 0; i < p.nagents; i++)
                if (strcmp(p.agents[i].name, script_name) == 0) sagent = i;
            if (sagent < 0) {
                fprintf(stderr, "hunch-vm: unknown agent '%s'\n", script_name);
                return 1;
            }
        }
        if (script_actions) {
            const char *q = script_actions;
            sacts = (int *)xmalloc(sizeof(int) * (strlen(q) + 1));
            while (*q) {
                sacts[sn++] = atoi(q);
                while (*q && *q != ',') q++;
                if (*q == ',') q++;
            }
        }

        FILE *out = stdout;
        if (outpath) {
            out = fopen(outpath, "w");
            if (!out) die("cannot open trace output");
        }
        emit_meta(out, &p);

        int episodes = (int)arg_long(argc, argv, "--episodes", 1);
        int want_forks = has_flag(argc, argv, "--forks");
        int strict = has_flag(argc, argv, "--strict");
        for (int e = 0; e < episodes; e++) {
            if (e) fputc(',', out);
            /* The VM starts fresh each episode, but the report covers them all. */
            Notes seen = vm.notes;
            vm_init(&vm, &p, &s);
            vm.notes = seen;
            vm.strict = strict;
            vm.gamma = cfg.gamma;
            vm.script_agent = sagent;
            vm.script_n = sn;
            vm.script_pos = 0;
            vm.script = sacts;
            vm.ask = has_flag(argc, argv, "--ask");
            vm.tab = pol.tab;
            vm.tab_mode = pol.tab ? TAB_AVERAGE : TAB_OFF;
            vm.search = pol.search;
            if (pol.search) pol.search->horizon = horizon;
            trace_episode(&p, nets, seed + (uint64_t)e * 7919ULL, horizon, out,
                          want_forks, &vm, &s);
        }
        fputs("]}", out);
        if (out != stdout) fclose(out);
        if (strict) vm_report(&vm, stderr);
        policy_free(&pol);
        free(sacts);
        return 0;
    }

    if (strcmp(cmd, "eval") == 0 || strcmp(cmd, "speed") == 0) {
        /* A search is not networks: the first agent's mean return with every
         * agent searching, on the episodes the other kinds are scored on.
         * Every decision is a whole search, so ask for few episodes. */
        Policy spol;
        memset(&spol, 0, sizeof(spol));
        if (polpath && policy_load(polpath, &p, &spol) && spol.search) {
            int episodes = (int)arg_long(argc, argv, "--episodes", cfg.eval_episodes);
            spol.search->horizon = horizon;
            Eval ev;
            double t0 = now_sec();
            eval_any(&p, &spol, 0, NULL, episodes, seed, horizon, &ev);
            double wall = now_sec() - t0;
            if (strcmp(cmd, "speed") == 0)
                printf("{\"episodes\":%d,\"wall_sec\":%.4f,\"return\":%.6f,\n"
                       "\"episodes_per_sec\":%.1f,\"steps_per_sec\":%.0f}\n", episodes, wall,
                       ev.mean[0], (double)episodes / wall, (double)episodes * horizon / wall);
            else
                printf("{\"episodes\":%d,\"return\":%.9g}\n", episodes, ev.mean[0]);
            policy_free(&spol);
            return 0;
        }
        policy_free(&spol);
        Net *nets[MAX_AGENTS];
        for (int i = 0; i < MAX_AGENTS; i++) nets[i] = NULL;
        for (int i = 0; i < p.nagents; i++)
            nets[i] = net_new(p.agents[i].obs_dim, p.agents[i].n_act, seed);
        if (polpath && !load_policy(polpath, &p, nets)) {
            fprintf(stderr, "hunch-vm: could not read policy %s\n", polpath);
            return 1;
        }
        int fz_agent = -1, fz_action = 0;
        {
            const char *fz = arg_value(argc, argv, "--freeze", NULL);
            if (fz) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%s", fz);
                char *colon = strchr(buf, ':');
                if (!colon) { fprintf(stderr, "hunch-vm: --freeze wants NAME:ACTION\n"); return 1; }
                *colon = 0;
                fz_action = atoi(colon + 1);
                for (int i = 0; i < p.nagents; i++)
                    if (strcmp(p.agents[i].name, buf) == 0) fz_agent = i;
                if (fz_agent < 0) { fprintf(stderr, "hunch-vm: unknown agent '%s'\n", buf); return 1; }
            }
        }

        int episodes = (int)arg_long(argc, argv, "--episodes",
                                     strcmp(cmd, "speed") == 0 ? 2000 : cfg.eval_episodes);
        if (strcmp(cmd, "speed") == 0) {
            double t0 = now_sec();
            double r = eval_policy_frozen(&p, nets, episodes, seed, horizon, 0,
                                          fz_agent, fz_action);
            double wall = now_sec() - t0;
            printf("{\"episodes\":%d,\"wall_sec\":%.4f,\"return\":%.6f,\n", episodes, wall, r);
            printf("\"episodes_per_sec\":%.1f,\"steps_per_sec\":%.0f}\n",
                   (double)episodes / wall, (double)episodes * horizon / wall);
        } else {
            double r = eval_policy_frozen(&p, nets, episodes, seed, horizon, 0,
                                          fz_agent, fz_action);
            printf("{\"episodes\":%d,\"return\":%.9g}\n", episodes, r);
        }
        for (int i = 0; i < MAX_AGENTS; i++) net_free(nets[i]);
        return 0;
    }

    usage();
    return 1;
}
#endif
