from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        mod = 10**9 + 7
        # Map each end index to its required inversion count
        req = {end: cnt for end, cnt in requirements}
        
        # dp[i] will hold number of ways to build a permutation of size current_size
        # with a given inversion count (index in the list).
        dp = [1]  # for size 0, only inversion count 0 is possible
        max_inv = 0  # maximum inversion count for current dp
        
        for i in range(0, n):
            # Build prefix sums of dp for fast range-sum queries
            pre = dp[:]  # copy
            for j in range(1, len(pre)):
                pre[j] = (pre[j-1] + pre[j]) % mod
            
            # New maximum inversion count after inserting element i
            next_max_inv = max_inv + i
            dp2 = [0] * (next_max_inv + 1)
            
            # Transition: when inserting the new largest element (i),
            # inserting at position p (0 <= p <= i) adds p new inversions.
            # So dp2[k] = sum_{p=0..i} dp[k-p], i.e., sum dp[t] for t in [k-i .. k]
            for k in range(next_max_inv + 1):
                low = k - i
                if low < 0:
                    low = 0
                high = k
                if high > max_inv:
                    high = max_inv
                if low <= high:
                    res = pre[high]
                    if low > 0:
                        res = res - pre[low - 1]
                    # bring back to [0, mod)
                    if res < 0:
                        res += mod
                    else:
                        res %= mod
                    dp2[k] = res
            
            # If there's a requirement on the prefix ending at i (i.e. size i+1),
            # we must have exactly req[i] inversions. Zero out all other counts.
            if i in req:
                needed = req[i]
                # If the needed count is outside the achievable range, answer is 0
                if needed < 0 or needed > next_max_inv:
                    return 0
                val = dp2[needed]
                # zero out and keep only the needed count
                dp2 = [0] * (next_max_inv + 1)
                dp2[needed] = val
            
            # Move to next step
            dp = dp2
            max_inv = next_max_inv
        
        # After inserting all n elements, dp holds counts for permutations of size n.
        # Because the problem guarantees a requirement for end = n-1, dp is already
        # filtered to a single non-zero entry. We sum just in case.
        return sum(dp) % mod