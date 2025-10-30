class Solution:
    def maximumLength(self, s: str) -> int:
        max_length = -1
        unique_chars = set(s)

        for char in unique_chars:
            n = len(s)
            max_consecutive = 0
            current_consecutive = 0
            for c in s:
                if c == char:
                    current_consecutive += 1
                else:
                    max_consecutive = max(max_consecutive, current_consecutive)
                    current_consecutive = 0
            max_consecutive = max(max_consecutive, current_consecutive)

            for length in range(max_consecutive, 0, -1):
                sub = char * length
                count = 0
                for i in range(n - length + 1):
                    if s[i:i+length] == sub:
                        count += 1

                if count >= 3:
                    max_length = max(max_length, length)
                    break

        return max_length