"""
配置加载模块（YAML + 环境变量）。

设计目标：
1. 规则、敏感词、哲学家角色均放在独立 YAML 文件中，运行时动态加载；
2. 默认配置路径为仓库根目录下的 config/；
3. 保持简单：不引入复杂依赖与过度工程化。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    """
    推断仓库根目录。

    约定：debate_arena/ 与 config/ 同级放置在仓库根目录下。
    """

    here = Path(__file__).resolve()
    return here.parent.parent


@dataclass(frozen=True)
class ConfigPaths:
    """
    配置文件路径集合。

    通过集中管理路径，避免后续模块各自拼路径造成维护成本上升。
    """

    root: Path

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def chairman_rules(self) -> Path:
        return self.config_dir / "chairman_rules.yaml"

    @property
    def debate_rules(self) -> Path:
        return self.config_dir / "debate_rules.yaml"

    @property
    def sensitive_keywords(self) -> Path:
        return self.config_dir / "sensitive_keywords.yaml"

    @property
    def clerk_rules(self) -> Path:
        return self.config_dir / "clerk_rules.yaml"

    @property
    def west_roles_dir(self) -> Path:
        return self.config_dir / "philosopher_roles" / "west"

    @property
    def east_roles_dir(self) -> Path:
        return self.config_dir / "philosopher_roles" / "east"

    def role_file(self, side: str, name: str) -> Path:
        """
        获取指定阵营(side)与角色名(name)对应的 YAML 文件路径。

        side:
          - west: 西方哲学家
          - east: 东方哲学家
        name:
          - 不含扩展名（例如 socrates / confucius）
        """

        if side not in {"west", "east"}:
            raise ValueError(f"未知 side: {side}")
        roles_dir = self.west_roles_dir if side == "west" else self.east_roles_dir
        return roles_dir / f"{name}.yaml"


def default_paths() -> ConfigPaths:
    """
    获取默认配置路径集合。
    """

    return ConfigPaths(root=_repo_root())


def load_yaml(path: Path) -> dict[str, Any]:
    """
    读取 YAML 文件并返回 dict。

    说明：
    - 若 YAML 为空，则返回空 dict；
    - 若文件不存在，抛出 FileNotFoundError，让上层明确感知配置缺失。
    """

    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是映射(dict)，但实际为: {type(data)} ({path})")
    return data


def load_chairman_rules(paths: ConfigPaths | None = None) -> dict[str, Any]:
    """
    加载主席规则（强规则引擎的阈值、模板、评分权重等）。
    """

    p = paths or default_paths()
    return load_yaml(p.chairman_rules)


def load_debate_rules(paths: ConfigPaths | None = None) -> dict[str, Any]:
    """
    加载辩手通用规则（结构、哲学性、文学性、输出字段等）。
    """

    p = paths or default_paths()
    return load_yaml(p.debate_rules)


def load_sensitive_keywords(paths: ConfigPaths | None = None) -> dict[str, Any]:
    """
    加载敏感内容触发词表。
    """

    p = paths or default_paths()
    return load_yaml(p.sensitive_keywords)


def load_clerk_rules(paths: ConfigPaths | None = None) -> dict[str, Any]:
    """
    加载书记员规则（文学化整理提示词与输出规范）。
    """

    p = paths or default_paths()
    return load_yaml(p.clerk_rules)


def load_role(side: str, name: str, paths: ConfigPaths | None = None) -> dict[str, Any]:
    """
    加载指定哲学家角色配置。
    """

    p = paths or default_paths()
    return load_yaml(p.role_file(side=side, name=name))
