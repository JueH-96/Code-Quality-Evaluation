from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        # Explanation:
        # We are allowed to perform one operation: select a contiguous subarray S and add an integer x to every
        # element in S. After that, we want to maximize the count of elements equal to k.
        #
        # Notice that if an index i is not chosen (i.e. i not in S), then its value remains unchanged.
        # If an index i in S is to become k (after adding x) we must have:
        #    nums[i] + x == k  =>  x == k - nums[i].
        # However, the same x must work for all indices in S.
        # Thus, if we want some indices in S to convert to k, they must “agree” on the value of x.
        # In other words, if we choose a candidate value v (v != k) and set x = k - v,
        # then some indices in S that originally equal v will convert to k.
        #
        # But note: S may also contain indices that are already k or that are not v;
        # after adding x, these indices will not remain equal to k (unless in the trivial case v == k when x==0).
        #
        # So, when choosing S we have two contributions:
        #   1. Those indices outside S remain unchanged. So if an index is originally k (and not in S),
        #      it still contributes to the frequency of k.
        #   2. Within S, only those indices i with nums[i] = v will convert to k when we use x = k - v.
        #
        # Therefore, if we choose a candidate v (v != k) and a contiguous block S, then
        # the overall frequency of k becomes:
        #       frequency = (# outside S that are originally k) + (# inside S with value v)
        #
        # Let base = total number of indices where nums[i] == k (if left untouched).
        # If S covers some positions, then we lose the ones in S where nums[i]==k (because when adding x they become k + x = 2k-v ≠ k)
        #
        # The net gain from S is:
        #      (count of indices in S where nums[i]==v) - (count of indices in S where nums[i]==k).
        # So overall frequency = base + gain.
        #
        # Our task is to choose a candidate v and a contiguous subarray S that maximizes:
        #         gain = (# indices in S with value v) - (# indices in S with value k)
        #
        # We must perform the operation once. However, if no beneficial subarray exists, we may pick an S that yields no gain.
        #
        # We can compute this using a variation of the maximum subarray sum problem (Kadane’s algorithm)
        # on a "difference array" built for candidate v as follows:
        #   For each index i:
        #       diff[i] = 1   if nums[i] == v   (we want to count these)
        #                 -1  if nums[i] == k   (we lose these if they are in S)
        #                 0   otherwise.
        #
        # Then, for candidate v (v ≠ k), the best gain is the maximum sum over any contiguous subarray of diff.
        # We compute this maximum gain; if it is negative, we can simply choose S to have a gain of 0.
        #
        # We do this for each candidate v from 1 to 50 (except v == k).
        # Finally, the best overall frequency is:
        #         ans = base + max(0, best_gain_over_all_candidates)
        #
        # Note: If we choose candidate v == k (which would mean x==0), then S would not change anything, and the frequency remains base.
        #
        # Complexity: O(50*n), which is acceptable for n up to 1e5.
        
        base = sum(1 for num in nums if num == k)
        ans = base  # worst-case, no beneficial operation
        # Try every possible candidate value v (except k) to “convert” to k.
        for v in range(1, 51):
            if v == k:
                continue
            cur_gain = 0
            best_gain = 0  # we allow not picking any subarray if all choices are negative
            for num in nums:
                # Compute contribution at this index:
                # If num == v, then if we add x = k - v this index becomes k => contribution +1.
                # If num == k, then this index (if in S) will become k + (k-v) = 2k - v ≠ k, so we lose one occurrence => -1.
                # Else: 0.
                if num == v:
                    delta = 1
                elif num == k:
                    delta = -1
                else:
                    delta = 0
                cur_gain = max(delta, cur_gain + delta)
                best_gain = max(best_gain, cur_gain)
            ans = max(ans, base + best_gain)
        return ans


# The following is a simple driver to run the code.
if __name__ == '__main__':
    import sys
    input_data = sys.stdin.read().strip().split()
    # We assume input is given in two lines:
    # First line: space separated integers for nums
    # Second line: integer k
    if not input_data:
        sys.exit(0)
    # Try to reconstruct lines using sys.stdin.read() is already used above;
    # so here we'll assume the first tokens belong to the nums list, and the last token is k.
    # This simple parser assumes that the final token is k.
    k = int(input_data[-1])
    nums = list(map(int, input_data[:-1]))
    
    sol = Solution()
    result = sol.maxFrequency(nums, k)
    sys.stdout.write(str(result))