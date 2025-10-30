class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        if n == 1:
            return len(words[0])

        dp = {(words[0][0], words[0][-1]): len(words[0])}

        for i in range(1, n):
            new_dp = {}
            current_word = words[i]
            for (start, end), length in dp.items():
                # join(str_{i-1}, words[i])
                new_end = current_word[-1]
                new_length = length + len(current_word)
                if end == current_word[0]:
                    new_length -= 1
                if (start, new_end) not in new_dp or new_length < new_dp[(start, new_end)]:
                    new_dp[(start, new_end)] = new_length

                # join(words[i], str_{i-1})
                new_start = current_word[0]
                new_length = len(current_word) + length
                if current_word[-1] == start:
                    new_length -= 1
                if (new_start, end) not in new_dp or new_length < new_dp[(new_start, end)]:
                    new_dp[(new_start, end)] = new_length
            dp = new_dp

        return min(dp.values())