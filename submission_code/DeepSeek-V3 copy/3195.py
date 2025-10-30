class Solution:
    def minimumSteps(self, s: str) -> int:
        total = 0
        black_count = 0
        for i in range(len(s)):
            if s[i] == '1':
                black_count += 1
            else:
                total += black_count
        return total