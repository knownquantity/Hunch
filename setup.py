"""Only here to ship the C runtime and the example worlds inside the package;
everything else is in pyproject.toml."""

import glob
import os
import re
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class build_with_runtime(build_py):
    def run(self):
        super().run()
        runtime = os.path.join(self.build_lib, "hunch", "_runtime")
        os.makedirs(runtime, exist_ok=True)
        shutil.copy(os.path.join("runtime", "hunch_vm.c"), runtime)
        worlds = os.path.join(self.build_lib, "hunch", "_worlds")
        os.makedirs(worlds, exist_ok=True)
        for path in glob.glob(os.path.join("worlds", "*.hunch")):
            shutil.copy(path, worlds)


def readme() -> str:
    """README.md for a page that is not in the repository: a link or an image
    given by a path would lead nowhere on PyPI, so each is pointed at GitHub."""
    text = open("README.md", encoding="utf-8").read()
    repo = "https://github.com/knownquantity/Hunch"

    def absolute(match):
        before, target, after = match.groups()
        if re.match(r"[a-z]+:|#", target):
            return match.group(0)
        kind = "raw" if re.search(r"\.(png|jpg|gif|svg|webp)$", target) else "blob"
        return f"{before}{repo}/{kind}/main/{target}{after}"

    text = re.sub(r"(\]\()([^)\s]+)(\))", absolute, text)
    return re.sub(r'((?:src|href)=")([^"]+)(")', absolute, text)


setup(cmdclass={"build_py": build_with_runtime},
      long_description=readme(), long_description_content_type="text/markdown")
