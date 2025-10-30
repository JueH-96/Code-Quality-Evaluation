class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        total = 0
        for i in range(n):
            d = int(s[i])
            if d == 0:
                continue
            if d in {1, 2, 5}:
                total += i + 1
            else:
                current_mod = 0
                pow_10 = 1
                count_inner = 0
                for j in range(i, -1, -1):
                    current_digit = int(s[j])
                    current_mod = (current_digit * pow_10 + current_mod) % d
                    pow_10 = (pow_10 * 10) % d
                    if current_mod == 0:
                        count_inner += 1
                total += count_inner
        return total