import unittest

from rag_app.chat.memory import WindowMemory


class WindowMemoryTests(unittest.TestCase):
    def test_only_last_n_turns_are_retained(self):
        memory = WindowMemory(max_turns=2)
        memory.add_turn("q1", "a1")
        memory.add_turn("q2", "a2")
        memory.add_turn("q3", "a3")
        self.assertEqual(memory.user_queries(), ["q2", "q3"])
        self.assertEqual(len(memory.messages()), 4)

    def test_clear_removes_every_message(self):
        memory = WindowMemory(max_turns=2)
        memory.add_turn("q", "a")
        memory.clear()
        self.assertEqual(memory.messages(), [])

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            WindowMemory(max_turns=0)

    def test_message_context_keeps_recent_complete_turns_within_budget(self):
        memory = WindowMemory(max_turns=10, max_chars=8)
        memory.add_turn("q1", "aaaa")
        memory.add_turn("q2", "bb")
        contents = [message.content for message in memory.messages()]
        self.assertEqual(contents, ["q2", "bb"])


if __name__ == "__main__":
    unittest.main()
