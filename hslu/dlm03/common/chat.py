"""Provides a chat abstraction."""

import json
from collections.abc import Iterable
from typing import IO, Any, AnyStr, Protocol

from hslu.dlm03.common import types


class ChatObserver(Protocol):
    """A protocol for chat observers."""

    def update(self, message: types.Message) -> None:
        """Updates the observer with the latest chat messages."""
        raise NotImplementedError


class Chat:
    """A class representing a chat."""

    _messages: list[types.Message]
    _observers: list[ChatObserver]

    def __init__(self, messages: Iterable[types.Message] | None = None) -> None:
        """Initializes a `Chat` instance.

        Args:
            messages: An iterable of messages to initialize the chat with.
        """
        self._messages = []
        self._observers = []
        if messages:
            self.append(*messages)

    def append(self, *messages: types.Message) -> None:
        """Appends messages to the chat.

        Args:
            *messages: The messages to append.
        """
        for message in messages:
            self._messages.append(message)
            for observer in self._observers:
                observer.update(message)

    def add_observer(self, observer: ChatObserver) -> bool:
        """Adds an observer to the chat.

        Args:
            observer: The observer to add.

        Returns:
            True if the observer was added, False otherwise.
        """
        if observer in self._observers:
            return False
        self._observers.append(observer)
        return True

    def remove_observer(self, observer: ChatObserver) -> bool:
        """Removes an observer from the chat.

        Args:
            observer: The observer to remove.

        Returns:
            True if the observer was removed, False otherwise.
        """
        if observer not in self._observers:
            return False
        self._observers.remove(observer)
        return True

    def serialize(self) -> bytes:
        """Serializes the chat to a string.

        Returns:
            A string representing the chat.
        """
        return json.dumps([types.message_to_dict(message) for message in self.messages]).encode()

    @classmethod
    def deserialize(cls, data: AnyStr, /) -> "Chat":
        """Deserializes a chat from a string.

        Args:
            data: The string to deserialize.
            *args: Positional arguments to pass to the Chat constructor.
            **kwargs: Keyword arguments to pass to the Chat constructor.

        Returns:
            A `Chat` instance.
        """
        messages = json.loads(data)
        return cls(messages=[types.dict_to_message(**message) for message in messages])

    def save(self, fp: IO[bytes]) -> None:
        """Saves the chat to a file-like object.

        Args:
            fp: The file-like object to save the chat to.
        """
        fp.write(self.serialize())

    @classmethod
    def load(cls, fp: IO[AnyStr], /, *args: Any, **kwargs: Any) -> "Chat":
        """Loads a chat from a file-like object.

        Args:
            fp: The file-like object to load the chat from.
            *args: Positional arguments to pass to the Chat constructor.
            **kwargs: Keyword arguments to pass to the Chat constructor.

        Returns:
            A `Chat` instance.
        """
        return cls.deserialize(fp.read(), *args, **kwargs)

    @property
    def messages(self) -> list[types.Message]:
        """Returns the messages in the chat.

        Returns:
            A list of `types.Message` objects.
        """
        return list(self._messages)
