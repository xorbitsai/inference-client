# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform
import sys
from sysconfig import get_config_vars

from pkg_resources import parse_version
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist

# From https://github.com/pandas-dev/pandas/pull/24274:
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == "darwin":
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system = platform.mac_ver()[0]
        python_target = get_config_vars().get(
            "MACOSX_DEPLOYMENT_TARGET", current_system
        )
        target_macos_version = "10.9"

        parsed_python_target = parse_version(python_target)
        parsed_current_system = parse_version(current_system)
        parsed_macos_version = parse_version(target_macos_version)
        if parsed_python_target <= parsed_macos_version <= parsed_current_system:
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = target_macos_version


repo_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(repo_root)


class ExtraCommandMixin:
    _extra_pre_commands = []

    def run(self):
        [self.run_command(cmd) for cmd in self._extra_pre_commands]
        super().run()

    @classmethod
    def register_pre_command(cls, cmd):
        cls._extra_pre_commands.append(cmd)


class CustomInstall(ExtraCommandMixin, install):
    pass


class CustomDevelop(ExtraCommandMixin, develop):
    pass


class CustomSDist(ExtraCommandMixin, sdist):
    pass


sys.path.append(repo_root)
versioneer = __import__("versioneer")


# build long description
def build_long_description():
    readme_path = os.path.join(os.path.abspath(repo_root), "README.md")

    with open(readme_path, encoding="utf-8") as f:
        return f.read()


def symlink_client():
    xinference_root = os.path.join(repo_root, "third_party", "inference", "xinference")
    dst_root = os.path.join(repo_root, "xinference_client")

    os.makedirs(os.path.join(dst_root, "client"), exist_ok=True)

    client_dir = os.path.join(xinference_root, "client", "restful")
    dst_client_dir = os.path.join(dst_root, "client", "restful")
    if os.path.exists(dst_client_dir):
        os.unlink(dst_client_dir)
    assert os.path.exists(client_dir)
    os.symlink(client_dir, dst_client_dir, target_is_directory=True)

    dst_init_file = os.path.join(dst_root, "client", "__init__.py")
    if os.path.exists(dst_init_file):
        os.remove(dst_init_file)
    # just create a new empty __init__.py
    f = open(dst_init_file, mode="w", encoding="utf-8")
    f.close()

    common_file = os.path.join(xinference_root, "client", "common.py")
    dst_common_file = os.path.join(dst_root, "client", "common.py")
    if os.path.exists(dst_common_file):
        os.unlink(dst_common_file)
    assert os.path.exists(common_file)
    os.symlink(common_file, dst_common_file, target_is_directory=False)

    types_file = os.path.join(xinference_root, "types.py")
    dst_types_file = os.path.join(dst_root, "types.py")
    if os.path.exists(dst_types_file):
        os.unlink(dst_types_file)
    assert os.path.exists(types_file)
    os.symlink(types_file, dst_types_file, target_is_directory=False)

    fields_file = os.path.join(xinference_root, "fields.py")
    dst_fields_file = os.path.join(dst_root, "fields.py")
    if os.path.exists(dst_fields_file):
        os.unlink(dst_fields_file)
    assert os.path.exists(fields_file)
    os.symlink(fields_file, dst_fields_file, target_is_directory=False)

    fields_file = os.path.join(xinference_root, "_compat.py")
    dst_fields_file = os.path.join(dst_root, "_compat.py")
    if os.path.exists(dst_fields_file):
        os.unlink(dst_fields_file)
    assert os.path.exists(fields_file)
    os.symlink(fields_file, dst_fields_file, target_is_directory=False)


setup_options = dict(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(
        {
            "install": CustomInstall,
            "develop": CustomDevelop,
            "sdist": CustomSDist,
        }
    ),
    long_description=build_long_description(),
    long_description_content_type="text/markdown",
)
symlink_client()
setup(**setup_options)
