class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        """
        We want a subsequence such that for consecutive indices i_j < i_{j+1},
        nums[i_{j+1}] - nums[i_j] >= i_{j+1} - i_j. 
        This can be rearranged to:
            nums[i_{j+1}] - i_{j+1} >= nums[i_j] - i_j.
        So any valid subsequence must be non-decreasing in the transformed value 
        A[i] = nums[i] - i.

        We aim to find the maximum sum of a subsequence subject to that ordering condition.
        It becomes a "maximum-sum non-decreasing subsequence" problem on A[i].
        
        Steps:
        1) Create an array A[i] = nums[i] - i.
        2) Coordinate-compress A (since it can be large).
        3) Use a Fenwick Tree (Binary Indexed Tree) or Segment Tree to maintain 
           the best (maximum) subsequence sum up to each compressed A-value.
        4) dp[i] = (max subsequence sum for all indices j<s.t.A[j] <= A[i]) + nums[i].
        5) The answer is max(dp[i]) for 0 <= i < n.
        """

        # Fenwick Tree (Binary Indexed Tree) for range max queries and updates
        # Typically Fenwicks are used for prefix sums; here, we store max values.
        class FenwickTree:
            def __init__(self, size):
                self.size = size
                self.tree = [-float("inf")] * (size+1)  # 1-based indexing

            def update(self, idx, val):
                while idx <= self.size:
                    self.tree[idx] = max(self.tree[idx], val)
                    idx += idx & -idx

            def query(self, idx):
                # max from 1..idx
                result = -float("inf")
                while idx > 0:
                    result = max(result, self.tree[idx])
                    idx -= idx & -idx
                return result

        n = len(nums)
        if n == 1:
            return nums[0]  # single element

        # Build A array
        A = [nums[i] - i for i in range(n)]

        # Coordinate-compress A
        # Get sorted unique values
        sorted_unique_A = sorted(set(A))
        # Create a mapping from value -> rank
        rank_map = {}
        for i, val in enumerate(sorted_unique_A):
            rank_map[val] = i + 1  # 1-based

        # Fenwick tree for "max subsequence sum" up to a rank
        fenwicksz = len(sorted_unique_A)
        fenwicksum = FenwickTree(fenwicksz)

        dp = [0]*n
        max_sum = -float("inf")

        for i in range(n):
            r = rank_map[A[i]]
            best_before = fenwicksum.query(r)  # best subsequence sum up to rank r
            if best_before == -float("inf"):
                # no valid subsequence found so far with A[j] <= A[i]
                dp[i] = nums[i]
            else:
                dp[i] = best_before + nums[i]
            # update fenwicksum at rank r with dp[i]
            fenwicksum.update(r, dp[i])
            max_sum = max(max_sum, dp[i])

        return max_sum