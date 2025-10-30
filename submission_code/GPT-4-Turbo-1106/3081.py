class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        # Since the array is sorted, we can remove pairs from the start and end
        # until we cannot find a valid pair (i.e., nums[i] < nums[j]).
        i, j = 0, len(nums) - 1
        while i < j:
            if nums[i] < nums[j]:
                i += 1
                j -= 1
            else:
                break
        # The remaining length of the array is the difference between the pointers
        # plus one because the array is 0-indexed.
        return j - i + 1