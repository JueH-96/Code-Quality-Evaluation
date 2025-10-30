from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        # Explanation:
        # We want to maximize the number of indices with value k after doing one contiguous operation. 
        #
        # The operation: choose a contiguous subarray S and an integer x. For every index i in S, we set:
        #    nums[i] = nums[i] + x.
        #
        # After the operation, the frequency of k equals:
        #    (all indices outside S that were originally k) +
        #    (for i in S, only those for which nums[i] + x == k).
        #
        # However, note that if an index i was originally k but lies in S and x != 0, then it becomes k+x ≠ k,
        # so we lose that occurrence. Thus picking S incurs a “penalty” if it covers some originally k’s.
        #
        # Now suppose we choose a value a and decide to “fix” (convert) all numbers with value a in our chosen subarray.
        # To turn a into k we need x = k - a. Then, within S:
        #    For any index i with nums[i] == a, after addition we get a + (k - a) = k.
        #    But for any index i with nums[i] == k (or any other number different from a) in S,
        #       after adding k - a, we get k + (k - a) or some other value – not k.
        #
        # Thus if we pick candidate number a (with a != k) and use x = k - a,
        # then only the indices in S where nums[i] == a will become k,
        # whereas any index originally k that gets included will be ruined.
        #
        # Let total_k be the count of k in the entire array (these remain k if not touched).
        # And let S be our chosen contiguous segment.
        # Then after the operation the frequency of k equals:
        #    total_k - (# of indices in S that were originally k) + (# of indices in S that were equal to a).
        #
        # Compared to doing nothing (yielding just total_k occurrences), the operation gives us a "net gain" of:
        #    (# of a's in S) - (# of k's in S).
        #
        # We are allowed to choose S arbitrarily, so for each candidate a (where a != k) we wish
        # to find a contiguous subarray S maximizing:
        #      gain(S, a) = (# of indices with value a) - (# of indices with value k)
        # (Indices with other values contribute 0.)
        #
        # Then the answer is:
        #      total_k + max(0, maximal gain over all candidates a)
        # (We take max with 0 because it might be optimal to not perform any effective conversion.)
        #
        # This can be done with a variant of Kadane's algorithm.
        # Since 1 <= nums[i] <= 50 (and k also in that range)
        # we can simply try each candidate a from 1 to 50 (skip a == k).
        
        total_k = nums.count(k)
        best_gain = 0
        n = len(nums)
        
        # Iterate over each possible candidate number a (different from k).
        for a in range(1, 51):
            if a == k:
                continue
            curr_sum = 0
            max_sum = 0
            for num in nums:
                # For each index, define the transformation value:
                #   If num == a, we can convert it to k (+1 gain).
                #   If num == k, then if included, we lose an occurrence (-1 penalty).
                #   Otherwise, no effect (0).
                if num == a:
                    val = 1
                elif num == k:
                    val = -1
                else:
                    val = 0
                
                # Standard Kadane update:
                curr_sum = max(val, curr_sum + val)
                max_sum = max(max_sum, curr_sum)
            best_gain = max(best_gain, max_sum)
        
        return total_k + best_gain

# Example Usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxFrequency([1,2,3,4,5,6], 1))            # Expected output: 2
    print(sol.maxFrequency([10,2,3,4,5,5,4,3,2,2], 10))   # Expected output: 4