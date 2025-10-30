from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pairCount = n // 2
        # The worst‐case is to change both numbers in every pair: cost = 2 per pair = n.
        # Now, for a given final constant X (0 <= X <= k), consider a pair (u, v).
        # We can get the desired absolute difference X with:
        # • 0 changes if |u-v| == X.
        # • 1 change if (even though |u-v| != X) we can change one number to some value that gives |new - other| = X.
        #   In fact, if we fix one number (say u) the possible differences we can “force” by changing the other
        #   are exactly those X for which one of u+X or u-X lies in [0, k]. Since u is in [0,k], this amounts
        #   to allowing any X such that X <= max(u, k-u). Similarly, by keeping v we allow all X such that
        #   X <= max(v, k-v). Overall, one change can fix the pair for any X in the range:
        #         X in [0, W]   where W = max( max(u, k-u), max(v, k-v) ).
        #
        # Thus for each pair, the “cost contribution” for a fixed X is:
        #   • 0 changes if X == |u-v|
        #   • 1 change if X is in [0, W] but X != |u-v| (because then one change can fix it).
        #   • Otherwise, we need 2 changes.
        #
        # So if we set baseline cost = 2 changes per pair (i.e. n), for each pair we can “save” 1 operation
        # if X is in [0, W], and a further 1 if X equals |u-v| (note: since |u-v| <= W always, the reduction is cumulative).
        #
        # Let:
        #   one_op[X] = number of pairs for which one change is enough when X is chosen; that is X falls in [0, W] for that pair.
        #   zero_op[X] = number of pairs that already have |u-v| == X.
        #
        # Then total changes for chosen X is:
        #   total_changes(X) = n - one_op[X] - zero_op[X]
        #
        # Our goal is to choose X (0 <= X <= k) that minimizes total_changes(X).
        
        # We use a difference-array technique to efficiently add +1 for all X in [0, W] for each pair.
        diff = [0] * (k + 2)
        zero = [0] * (k + 1)  # frequency for no-change possibility
        
        # Process each pair.
        for i in range(pairCount):
            u = nums[i]
            v = nums[n - i - 1]
            d = abs(u - v)
            # For no-change possibility: if X equals d, no change needed.
            zero[d] += 1
            # For one-change possibility: one change is enough if chosen X is in [0, W],
            # where W = max( max(u, k-u), max(v, k-v) ).
            W = max(max(u, k - u), max(v, k - v))
            diff[0] += 1
            # All X in [0, W] gain a saving of 1 op.
            if W + 1 <= k:
                diff[W + 1] -= 1
        
        # Build the one_op array from the difference array.
        one_op = [0] * (k + 1)
        curr = 0
        for X in range(k + 1):
            curr += diff[X]
            one_op[X] = curr
        
        # Baseline cost: 2 changes per pair = n
        ans = float('inf')
        for X in range(k + 1):
            # For each pair we save:
            # - 1 op if X is in the one-change range (one_op[X] many pairs)
            # - an additional 1 op for pairs already having difference X (zero[X])
            changes = n - one_op[X] - zero[X]
            if changes < ans:
                ans = changes
        return ans


# For local testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums = [1, 0, 1, 2, 4, 3]
    k_val = 4
    print(sol.minChanges(nums, k_val))  # Expected output: 2

    # Example 2:
    nums = [0, 1, 2, 3, 3, 6, 5, 4]
    k_val = 6
    print(sol.minChanges(nums, k_val))  # Expected output: 2