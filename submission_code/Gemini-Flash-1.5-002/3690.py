class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        if numOps >= n:
            return 1

        best_len = float('inf')
        for i in range(1 << n):
            temp_s = list(s)
            ops = 0
            for j in range(n):
                if (i >> j) & 1:
                    temp_s[j] = '1' if temp_s[j] == '0' else '0'
                    ops += 1

            if ops <= numOps:
                temp_s_str = "".join(temp_s)
                max_len = 0
                curr_len = 1
                for k in range(1, n):
                    if temp_s_str[k] == temp_s_str[k-1]:
                        curr_len += 1
                    else:
                        max_len = max(max_len, curr_len)
                        curr_len = 1
                max_len = max(max_len, curr_len)
                best_len = min(best_len, max_len)

        return best_len