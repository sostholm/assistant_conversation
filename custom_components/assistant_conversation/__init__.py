"""The OpenAI Conversation integration."""

from __future__ import annotations

import logging
from typing import Literal

import openai
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, MATCH_ALL
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import (
    config_validation as cv,
    intent,
    issue_registry as ir,
    selector,
    template,
)
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_ASSISTANT_URL,
    DEFAULT_ASSISTANT_URL,
    DOMAIN,
)

import aiohttp
import json

_LOGGER = logging.getLogger(__name__)
CALL_ASSISTANT = "call_assistant"

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""

    async def call_assistant(call: ServiceCall) -> ServiceResponse:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "messages": [call.data["prompt"]],
                }
                async with session.post(call.data["assistant_url"], data=json.dumps(payload)) as resp:
                    result = await resp.text()
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        return result

    hass.services.async_register(
        DOMAIN,
        CALL_ASSISTANT,
        call_assistant,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required("prompt"): cv.string,
                vol.Required("assistant_url"): cv.string,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    conversation.async_set_agent(hass, entry, OpenAIAgent(hass, entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    conversation.async_unset_agent(hass, entry)
    return True


class OpenAIAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        assistant_url = self.entry.options.get(CONF_ASSISTANT_URL, DEFAULT_ASSISTANT_URL)

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            messages = []
            conversation_id = ulid.ulid_now()

        messages.append({"role": "user", "content": user_input.text})

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "conversation_id": conversation_id,
                    "messages": messages,
                }
                async with session.post(assistant_url, data=json.dumps(payload)) as resp:
                    result = await resp.json()
        except Exception as err:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        _LOGGER.debug("Response %s", result)
        response = result['content']
        # messages.append(result)
        self.history[conversation_id] = messages

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _async_generate_prompt(self, raw_prompt: str) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
            },
            parse_result=False,
        )
