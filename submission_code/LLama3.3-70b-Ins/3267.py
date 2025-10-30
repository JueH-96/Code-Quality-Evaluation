class Solution:
    def maximumLength(self, s: str) -> int:
        max_length = -1
        for length in range(1, len(s) + 1):
            for i in range(len(s) - length + 1):
                substring = s[i:i + length]
                if len(set(substring)) == 1:  # Check if the substring is special
                    count = 0
                    for j in range(len(s) - length + 1):
                        if s[j:j + length] == substring:
                            count += 1
                    if count >= 3:
                        max_length = max(max_length, length)
        return max_length