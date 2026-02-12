from __future__ import annotations

import pathlib
from typing import Any

import yaml


class PromptBuilder:
    def __init__(self, prompt_config_path: str = "configs/prompts.yaml") -> None:
        cfg_path = pathlib.Path(prompt_config_path)
        self.config: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def build_train_prompt(self, merchant_text: str | None = None) -> str:
        text = self.config["train_prompt_template"].strip()
        constraints = self.config["json_constraints"].strip()
        if merchant_text:
            return f"{text}\n\n商家描述：{merchant_text}\n\n{constraints}"
        return f"{text}\n\n{constraints}"

    def build_infer_prompt(self, merchant_text: str | None = None) -> str:
        text = self.config["inference_prompt_template"].strip()
        constraints = self.config["json_constraints"].strip()
        if merchant_text:
            return f"{text}\n\n商家描述：{merchant_text}\n\n{constraints}"
        return f"{text}\n\n{constraints}"

    @property
    def system_prompt(self) -> str:
        return self.config["system_prompt"].strip()
