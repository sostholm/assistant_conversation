"""Config flow for OpenAI Conversation integration."""

from __future__ import annotations

import logging
import types
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    TemplateSelector,
)

from .const import (
    CONF_ASSISTANT_URL,
    DEFAULT_ASSISTANT_URL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_ASSISTANT_URL): str,
    }
)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_ASSISTANT_URL: DEFAULT_ASSISTANT_URL,
    }
)



class OpenAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ):
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}
        
        if user_input[CONF_ASSISTANT_URL] is None:
            errors["base"] = "missing_url"
        
        else:
            return self.async_create_entry(title="Assistant Conversation", data=user_input)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OpenAIOptionsFlow(config_entry)


class OpenAIOptionsFlow(OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="Assistant Conversation", data=user_input)
        schema = openai_config_option_schema(self.config_entry.options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def openai_config_option_schema(options: MappingProxyType[str, Any]) -> dict:
    """Return a schema for OpenAI completion options."""
    if not options:
        options = DEFAULT_OPTIONS
    return {
        vol.Required(
            CONF_ASSISTANT_URL,
            description={"suggested_value": "URL"},
        ): TemplateSelector()
    }
