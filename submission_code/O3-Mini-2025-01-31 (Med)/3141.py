from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        # Compute prefix sums for one copy of nums.
        # P[0] = 0, P[i] = nums[0] + ... + nums[i-1] for i from 1 to n.
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + nums[i]
        S = P[-1]  # Sum of the entire nums
        
        # Because nums has all positive elements,
        # P is strictly increasing. We build a dictionary: prefix value -> index.
        prefix_to_index = {val: i for i, val in enumerate(P)}
        
        INF = float('inf')
        ans = INF
        
        # ---------------------------------------------------------------------
        # Case 1: Subarray completely inside one copy (non-wrapping case)
        # In this case a contiguous subarray from index i to j (0 <= i < j <= n)
        # has sum = P[j] - P[i]. We search for pairs satisfying:
        #     P[j] - P[i] == target.
        for i in range(n+1):
            needed = P[i] + target  # We want some j with P[j] == needed.
            if needed in prefix_to_index:
                j = prefix_to_index[needed]
                if j > i:  # (target > 0 guarantees j > i when a solution exists)
                    candidate_length = j - i
                    if candidate_length < ans:
                        ans = candidate_length
        
        # ---------------------------------------------------------------------
        # Case 2: Subarray that “wraps” from the end of one copy into the beginning
        # of a later copy. In the infinite repetition, any subarray that crosses a cycle
        # boundary can be expressed as:
        #
        #    (S - P[i]) + q * S + P[j]
        #
        # where 0 <= i < n (starting from position i in one copy) and 0 <= j <= n
        # (ending at position j in the next copy), and q >= 0 counts whole extra cycles.
        # We require:
        # 
        #    (S - P[i] + P[j]) + q*S = target
        #
        # Let X = (S - P[i] + P[j]). Then target = X + q*S, so we must have
        #    X ≡ target  (mod S)
        # and q = (target - X) // S.
        #
        # Note that given P[i] in [0,S) (since i in [0,n)), and P[j] in [0,S]
        # (j in [0, n]), X = S - P[i] + P[j] can range at most up to 2*S.
        # In other words, for a valid pair (i,j) forming the “partial” piece we must have:
        #    X ∈ (0, 2*S] and X ≤ target.
        #
        # Since X ≡ target (mod S), there are at most two candidates for X in (0,2*S]:
        # Let r = target % S.
        #   • If r == 0, then X must equal S (since X = 0 is impossible as S-P[i] > 0).
        #   • If r > 0, then possible X values are: X = r, and possibly X = r+S
        #     (provided they are ≤ 2*S and ≤ target).
        cand_X = []
        r = target % S
        if r == 0:
            if S <= target and S <= 2 * S:
                cand_X.append(S)
        else:
            if r <= target and r <= 2 * S:
                cand_X.append(r)
            if r + S <= target and r + S <= 2 * S:
                cand_X.append(r + S)
        
        # For each candidate X, we have target = X + q * S, so:
        #    q = (target - X) // S   (and automatically q >= 0).
        # Also, we need to choose indices 0 <= i < n and 0 <= j <= n satisfying:
        #    S - P[i] + P[j] = X   -->   P[j] = X - S + P[i]
        # For each such pair, the overall subarray length is:
        #    (n - i) [from i to end of copy]  +  q*n (the complete cycles in between)
        #    + j (the first j elements from the next copy)
        for X in cand_X:
            q = (target - X) // S
            for i in range(n):
                needed = X - S + P[i]  # we need a j such that P[j] == needed.
                if needed in prefix_to_index:
                    j = prefix_to_index[needed]
                    candidate_length = (n - i) + j + q * n
                    if candidate_length < ans:
                        ans = candidate_length
        
        return ans if ans != INF else -1


# Testing with the provided examples:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # infinite_nums = [1, 2, 3, 1, 2, 3, 1, 2, ...] 
    # A subarray like [2, 3] gives a sum of 5 with length 2.
    print(sol.minSizeSubarray([1, 2, 3], 5))  # Expected output: 2

    # Example 2:
    # infinite_nums = [1, 1, 1, 2, 3, 1, 1, 1, 2, 3, ...]
    # A subarray like [2, 3] from indices 3 and 4 gives a sum of 5, but for target 4,
    # one optimal subarray is [1, 3] wrapping around with length 2.
    print(sol.minSizeSubarray([1, 1, 1, 2, 3], 4))  # Expected output: 2

    # Example 3:
    # infinite_nums = [2, 4, 6, 8, 2, 4, 6, 8, ...]
    # There is no contiguous subarray with sum equal to 3.
    print(sol.minSizeSubarray([2, 4, 6, 8], 3))  # Expected output: -1