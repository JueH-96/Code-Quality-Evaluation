class Solution:
    def countSubstrings(self, s: str) -> int:
        f = {d: [0] * d for d in range(1, 10)}
        ans = 0
        for char in s:
            d0 = int(char)
            new_f = {}
            for d_val in range(1, 10):
                state = [0] * d_val
                for r in range(d_val):
                    cnt = f[d_val][r]
                    if cnt:
                        new_r = (r * 10 + d0) % d_val
                        state[new_r] += cnt
                state[d0 % d_val] += 1
                new_f[d_val] = state
            if d0 != 0:
                ans += new_f[d0][0]
            f = new_f
        return ans