from typing import List
import bisect

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # For each index i, we want to know the first index to the right 
        # where an element strictly greater than nums[i] appears.
        # In any subarray starting at i, if we stop before that index,
        # no element greater than nums[i] appears.
        # Thus if we take any subarray [i, j] with j < nge[i] and with nums[j] == nums[i],
        # then the first and last elements equal nums[i] and because no inner element is > nums[i],
        # the maximum element in that subarray is nums[i].
        nge = [n] * n  # default: no greater element found, so use n (out of bound index)
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                nge[idx] = i
            stack.append(i)
        
        # Build a dictionary mapping each value to the sorted list of its indices.
        positions = {}
        for i, v in enumerate(nums):
            positions.setdefault(v, []).append(i)
        
        total = 0
        # For each index i, count subarrays [i, j] (with i <= j) that are valid.
        # Valid means: nums[i] == nums[j] and in the subarray, no element is greater than nums[i]. 
        # For fixed i with value v, if we choose any later index j with the same value v
        # and j < nge[i] (since nge[i] is where an element > v occurs), then [i,j] is valid.
        # Also, the subarray consisting only of nums[i] is valid.
        #
        # We use the positions list for value v to quickly count how many indices j (with j > i)
        # fall in the interval [i+1, nge[i]-1].
        for i, v in enumerate(nums):
            lst = positions[v]
            # Find the index of i in lst.
            pos_i = bisect.bisect_left(lst, i)
            # Find the first index in lst at which the array index is >= nge[i]
            pos_ng = bisect.bisect_left(lst, nge[i])
            # Among indices in lst, those between pos_i+1 and pos_ng-1 (inclusive)
            # represent valid endpoints j (with j > i and j < nge[i]).
            count_valid_j = max(0, pos_ng - pos_i - 1)
            # Add the valid pairs starting with i plus the single-element subarray [i].
            total += count_valid_j + 1
        
        return total

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    print(sol.numberOfSubarrays([1,4,3,3,2]))  # Expected output: 6
    print(sol.numberOfSubarrays([3,3,3]))      # Expected output: 6
    print(sol.numberOfSubarrays([1]))          # Expected output: 1