class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        parts, end, split_len = [word], word, len(word) // numFriends + 1
        while len(parts) < numFriends:
            parts, end = parts[:-1] + [end[-split_len:] + start for start, end in zip(parts[-1:], end[:-split_len])], end[split_len:]
        return min(parts, key = lambda item: item.ljust(len(word), 'a'), reverse=True)