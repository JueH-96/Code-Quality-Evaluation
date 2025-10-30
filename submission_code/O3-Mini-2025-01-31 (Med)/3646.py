MOD = 10**9 + 7

class Solution:
    def sumOfGoodSubsequences(self, nums: list[int]) -> int:
        # dp will hold for each value v a tuple:
        # (total number of good subsequences ending with value v, 
        #  total sum of elements over those subsequences)
        #
        # When processing a new element with value v, it can either:
        # 1. Start a new subsequence [v] (good by default).
        # 2. Extend any existing good subsequence that ends with v-1 or v+1.
        #
        # For an extension from a subsequence ending at 'adj' (v-1 or v+1):
        #   New subsequence's count contribution: same as the count ending with adj.
        #   New subsequence's sum contribution: previous sum plus v times the count,
        #     that is: dp[adj].sum + dp[adj].count * v.
        
        from collections import defaultdict

        dp = defaultdict(lambda: (0, 0))
        
        for v in nums:
            # Calculate extension opportunities from adjacent values.
            cnt_ext, sum_ext = 0, 0
            for adj in (v - 1, v + 1):
                if adj in dp:
                    c, s = dp[adj]
                    cnt_ext = (cnt_ext + c) % MOD
                    sum_ext = (sum_ext + s) % MOD
                    
            # New subsequence: just [v].
            # Extend from adjacent subsequences:
            new_cnt = (1 + cnt_ext) % MOD
            new_sum = (v + (sum_ext + v * cnt_ext) % MOD) % MOD
            
            # Update dp for value v: Note that we are adding subsequences ending at v
            # from all occurrences.
            old_cnt, old_sum = dp[v]
            dp[v] = ((old_cnt + new_cnt) % MOD, (old_sum + new_sum) % MOD)
        
        # The final answer is the sum of all subsequence sums stored in dp.
        ans = 0
        for _, subseq_sum in dp.values():
            ans = (ans + subseq_sum) % MOD
        return ans

# Example usage and simple tests:
if __name__ == '__main__':
    sol = Solution()
    print(sol.sumOfGoodSubsequences([1, 2, 1]))  # Output: 14
    print(sol.sumOfGoodSubsequences([3, 4, 5]))  # Output: 40