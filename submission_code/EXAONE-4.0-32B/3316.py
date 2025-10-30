import bisect

mod = 10**9 + 7

class Solution:
    def sumOfPowers(self, nums: list[int], k: int) -> int:
        arr = sorted(nums)
        n = len(arr)
        pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diff = arr[j] - arr[i]
                pairs.add(diff)
        distinct_ds = sorted(pairs, reverse=True)
        
        def compute_F(d):
            dp_prev = [1] * n
            for c_val in range(2, k + 1):
                prefix = [0] * n
                prefix[0] = dp_prev[0] % mod
                for i in range(1, n):
                    prefix[i] = (prefix[i - 1] + dp_prev[i]) % mod
                dp_curr = [0] * n
                for i in range(n):
                    val = arr[i] - d
                    pos = bisect.bisect_right(arr, val, 0, i)
                    j_max = pos - 1
                    if j_max >= 0:
                        dp_curr[i] = prefix[j_max]
                    else:
                        dp_curr[i] = 0
                dp_prev = dp_curr
            return sum(dp_prev) % mod
        
        total_ans = 0
        prev_F = 0
        for d in distinct_ds:
            F_d = compute_F(d)
            count = (F_d - prev_F) % mod
            total_ans = (total_ans + count * d) % mod
            prev_F = F_d
        
        return total_ans % mod