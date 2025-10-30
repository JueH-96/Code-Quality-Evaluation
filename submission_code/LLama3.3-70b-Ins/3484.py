class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        for i in range(len(s) - 1):
            # Check if the current digit and the next digit have the same parity
            if (int(s[i]) % 2 == 0 and int(s[i + 1]) % 2 == 0) or (int(s[i]) % 2 != 0 and int(s[i + 1]) % 2 != 0):
                # If they have the same parity, swap them if the current digit is greater than the next digit
                if s[i] > s[i + 1]:
                    s[i], s[i + 1] = s[i + 1], s[i]
                    # After swapping, break the loop because we can only swap once
                    break
        return ''.join(s)