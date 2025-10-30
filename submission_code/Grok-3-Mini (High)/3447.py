class Solution:
    def clearDigits(self, s: str) -> str:
        chars = list(s)
        while True:
            digit_idx = -1
            for i in range(len(chars)):
                if chars[i].isdigit():
                    digit_idx = i
                    break
            if digit_idx == -1:
                break
            left_idx = -1
            for j in range(digit_idx - 1, -1, -1):
                if not chars[j].isdigit():
                    left_idx = j
                    break
            # Delete higher index first to avoid index shifting issues
            del chars[digit_idx]
            del chars[left_idx]
        return ''.join(chars)