from bisect import bisect_left
MOD = 10**9+7

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        # all possible differences a[j]-a[i] for i<j
        ds = set()
        for i in range(n):
            for j in range(i+1, n):
                ds.add(nums[j] - nums[i])
        ds = sorted(ds)
        
        # For each d in ds, compute C[d] = number of k-subsets with
        # min adjacent diff >= d.
        C = []
        for d in ds:
            # dp_prev[i] = # ways to pick t-1 elements ending at i
            # For t=1, dp_prev[i]=1
            dp_prev = [1]*n
            for t in range(2, k+1):
                # build prefix sums of dp_prev
                pref = [0]*(n+1)
                for i in range(n):
                    pref[i+1] = (pref[i] + dp_prev[i]) % MOD
                dp_cur = [0]*n
                # for each i, sum dp_prev[j] for j in [L..i-1] where nums[i]-nums[j]>=d
                for i in range(n):
                    # find smallest j so that nums[j] >= nums[i]-d
                    # we need nums[i]-nums[j] >= d  => nums[j] <= nums[i]-d
                    # so j_max = rightmost index with nums[j] <= nums[i]-d
                    # L = bisect_left(nums, nums[i]-d)
                    L = bisect_left(nums, nums[i] - d + 1)
                    # valid j are in [0..i-1] ∩ [0..L-1] => [0..L-1] if L-1 < i
                    if L <= i-1:
                        dp_cur[i] = (pref[L] if L<=0 else pref[L])  # pref[L] sums j in [0..L-1]
                        # But we want j in [0..i-1] and nums[i]-nums[j]>=d
                        # Actually L = first j with nums[j] >= nums[i]-d+1 
                        # so j<=L-1 satisfy nums[j] <= nums[i]-d
                        # sum over j=0..min(i-1,L-1) => pref[min(i-1,L-1)+1]
                        jmax = min(i-1, L-1)
                        dp_cur[i] = pref[jmax+1]
                dp_prev = dp_cur
            C.append(sum(dp_prev) % MOD)
        
        # Now compute sum of minimums:
        # count of subsets with exact min = C[i] - C[i+1]
        ans = 0
        for i, d in enumerate(ds):
            cnt = C[i]
            if i+1 < len(ds):
                cnt = (cnt - C[i+1]) % MOD
            ans = (ans + cnt * d) % MOD
        return ans