class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for t in range(1, n + 1):
            if self.is_possible(word, k, t):
                return t
        return -1  # According to the problem statement, there is always a solution.

    def is_possible(self, s: str, k: int, t: int) -> bool:
        n = len(s)
        current = s
        for _ in range(t):
            if len(current) < k:
                return False
            # The preserved part after removing k characters
            preserved = current[k:]
            # The added part must be such that after t steps, the string is s
            # We need to check if the preserved part is a prefix of the remaining part of s
            # and the added part is the next part of s
            pos = 0
            valid = True
            for _ in range(t):
                if len(current) < k:
                    valid = False
                    break
                preserved_part = current[k:]
                # Check if the preserved part matches the corresponding part in s
                if s[pos:pos + len(preserved_part)] != preserved_part:
                    valid = False
                    break
                pos += len(preserved_part)
                # The added part is the next k characters from s
                if pos + k > n:
                    valid = False
                    break
                added_part = s[pos:pos + k]
                current = preserved_part + added_part
                pos += k
            if valid and pos == n and current == s:
                return True
            current = s  # Reset for next t
        return False