from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        """
        Idea
        ----
        For a subsequence ( … ,j , i ) to be *balanced* we need  
                   nums[i] - nums[j] >= i - j
        ⇔     (nums[i] - i)  >=  (nums[j] - j)             (just rearrange)
        
        Thus a subsequence is balanced **iff** the sequence
                  key(k) = nums[k] - k
        is non-decreasing along the chosen indices.

        So we have the classical problem:

           “Given points with value  key(k)  and weight  nums[k],
            pick a non-decreasing subsequence in key that maximises
            the total weight.”

        Let
              dp[i]  = best sum of a balanced subsequence that *ends* in i
        then
              dp[i] = nums[i] + max_{ j<i , key(j) ≤ key(i) } dp[j] ,
        and when no such j exists we simply take the single element: dp[i]=nums[i].

        To compute the `max` efficiently we:
          • coordinate-compress all keys = nums[i]-i (≤10⁵ different values)  
          • maintain a Fenwick / BIT that, for every compressed key, stores
            the maximum dp seen so far.
          • for every i  
                best = query_BIT(prefix ≤ key(i))  
                dp[i] = nums[i] + max(0,best)      # ignore negative sums
                update_BIT(key(i), dp[i])

        Complexity
        ----------
        n ≤ 10⁵, BIT operations are O(log n):
            Time :  O(n log n)
            Memory: O(n)

        """
        n = len(nums)
        # -----  coordinate compression of keys  -----
        keys = [nums[i] - i for i in range(n)]
        sorted_unique = sorted(set(keys))
        comp = {v: idx+1 for idx, v in enumerate(sorted_unique)}  # 1-based for BIT
        m = len(sorted_unique)

        # -----  Fenwick tree for range maximum  -----
        INF_NEG = -10**20  # sentinel smaller than any possible sum

        bit = [INF_NEG]*(m+2)

        def bit_update(idx: int, val: int) -> None:
            while idx <= m:
                if val > bit[idx]:
                    bit[idx] = val
                idx += idx & -idx

        def bit_query(idx: int) -> int:          # maximum on [1 .. idx]
            res = INF_NEG
            while idx > 0:
                if bit[idx] > res:
                    res = bit[idx]
                idx -= idx & -idx
            return res

        # -----  main DP loop  -----
        answer = -10**18
        for i, num in enumerate(nums):
            k = comp[keys[i]]
            best_prev = bit_query(k)              # best dp with key ≤ current
            if best_prev < 0:
                best_prev = 0                    # start new if previous sums are negative
            dp_i = num + best_prev
            if dp_i > answer:
                answer = dp_i
            bit_update(k, dp_i)

        return answer