class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        total = 0
        c0_max = 200  # Since c0^2 <= 40000 and n <= 40000

        for i in range(n):
            c0 = 0
            c1 = 0
            for j in range(i, min(n, i + c0_max * 2)):  # Limit the inner loop to optimize
                if s[j] == '0':
                    c0 += 1
                else:
                    c1 += 1
                if c1 >= c0 * c0:
                    total += 1
                if c0 > c0_max:
                    break
        return total