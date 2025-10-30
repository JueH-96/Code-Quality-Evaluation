from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        # n is small (<=10), queries up to 1000, and values are small.
        # We can think about each index separately.
        # For index j, we need to decide which queries (among the ones whose range covers j) will be applied
        # to subtract exactly nums[j] in total.
        # Since each query decision is independent per index, the existence of a valid choice is equivalent
        # to the subset sum feasibility: Is there a subset (of query contributions v) among those queries that affect index j
        # that sums exactly to nums[j].
        # We'll process queries in order and use a bitset dynamic programming approach.
        # dp[j] will be an integer bitmask representing achievable sums for element j.
        #
        # We initialize for each j: only sum 0 is achievable.
        # For every query i in order (with query parameters: l, r, val):
        #   For every index j in [l, r], update dp[j] as:
        #       dp[j] = dp[j] OR (dp[j] shifted left by val)
        # After processing each query (i.e. after updating the relevant dp[j]), we check if for every j,
        #   the bit corresponding to nums[j] is set in dp[j].
        # If yes, we return the index+1 (i.e., number of queries processed).
        # If after processing all queries, one or more indices do not have dp[j] that can reach exactly nums[j],
        #   return -1.
        #
        # Note: Because our dp uses a bitmask, and as nums[j] can be up to 1000 and each query decrement up to 10,
        # the maximum achievable sum might be a bit large. However, since queries length is at most 1000,
        # and bit shifting integers in Python can be done quite efficiently, this approach is acceptable.
        
        n = len(nums)
        m = len(queries)
        
        # Initialize dp for each index j as bitset with 0-th bit set.
        dp = [1 for _ in range(n)]  # 1 represents that sum 0 is achievable.
        
        # Check initially, if all nums are 0 already.
        if all(num == 0 for num in nums):
            return 0
        
        # Process queries one by one.
        for i in range(m):
            l, r, val = queries[i]
            # For every index in the range [l, r], update its dp bitmask.
            for j in range(l, r+1):
                # dp[j] already contains all sums achievable using some selection of previous queries.
                # For every achievable sum s, an additional option is to include this query’s decrement (if it doesn't exceed required sum).
                # We can add a new achievable sum s+val. So we set bits that are achieved by (dp[j] shifted left by val).
                dp[j] |= (dp[j] << val)
            
            # After processing this query, check if every index j can achieve exactly nums[j].
            all_possible = True
            for j in range(n):
                if (dp[j] >> nums[j]) & 1 == 0:
                    all_possible = False
                    break
            if all_possible:
                return i + 1  # queries are 0-indexed so return count.
        
        return -1

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums = [2, 0, 2]
    queries = [[0, 2, 1], [0, 2, 1], [1, 1, 3]]
    print(sol.minZeroArray(nums, queries))  # Expected output: 2

    # Example 2:
    nums = [4,3,2,1]
    queries = [[1,3,2],[0,2,1]]
    print(sol.minZeroArray(nums, queries))  # Expected output: -1

    # Example 3:
    nums = [1,2,3,2,1]
    queries = [[0,1,1],[1,2,1],[2,3,2],[3,4,1],[4,4,1]]
    print(sol.minZeroArray(nums, queries))  # Expected output: 4

    # Example 4:
    nums = [1,2,3,2,6]
    queries = [[0,1,1],[0,2,1],[1,4,2],[4,4,4],[3,4,1],[4,4,5]]
    print(sol.minZeroArray(nums, queries))  # Expected output: 4