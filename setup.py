"""Only here to ship the C runtime and the example worlds inside the package;
everything else is in pyproject.toml."""

import glob
import os
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


setup(cmdclass={"build_py": build_with_runtime})
