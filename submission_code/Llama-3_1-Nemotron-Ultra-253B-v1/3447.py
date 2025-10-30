class Solution:
    def clearDigits(self, s: str) -> str:
        s = list(s)
        while True:
            first_digit_idx = None
            for i in range(len(s)):
                if s[i].isdigit():
                    first_digit_idx = i
                    break
            if first_digit_idx is None:
                break
            left_non_digit_idx = None
            for j in range(first_digit_idx - 1, -1, -1):
                if not s[j].isdigit():
                    left_non_digit_idx = j
                    break
            del s[first_digit_idx]
            del s[left_non_digit_idx]
        return ''.join(s)