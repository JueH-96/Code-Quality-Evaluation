class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        mod = 10**9 + 7
        dp_now = [[[0] * (limit + 1) for _ in range(2)] for __ in range(zero + 1)]
        
        if zero >= 1:
            dp_now[1][0][1] = 1
        if one >= 1:
            dp_now[0][1][1] = 1
        
        total_elements = zero + one
        for step in range(1, total_elements):
            dp_next = [[[0] * (limit + 1) for _ in range(2)] for __ in range(zero + 1)]
            for z in range(zero + 1):
                o = step - z
                if o < 0 or o > one:
                    continue
                for last in range(2):
                    for c in range(1, limit + 1):
                        ways = dp_now[z][last][c]
                        if ways == 0:
                            continue
                        if z < zero:
                            if last == 0:
                                if c < limit:
                                    dp_next[z + 1][0][c + 1] = (dp_next[z + 1][0][c + 1] + ways) % mod
                            else:
                                dp_next[z + 1][0][1] = (dp_next[z + 1][0][1] + ways) % mod
                        if o < one:
                            if last == 1:
                                if c < limit:
                                    dp_next[z][1][c + 1] = (dp_next[z][1][c + 1] + ways) % mod
                            else:
                                dp_next[z][1][1] = (dp_next[z][1][1] + ways) % mod
            dp_now = dp_next
        
        ans = 0
        for last in range(2):
            for c in range(1, limit + 1):
                ans = (ans + dp_now[zero][last][c]) % mod
        return ans