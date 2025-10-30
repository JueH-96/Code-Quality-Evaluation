class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        count = 0
        m = len(pattern)  # pattern length
        n = len(nums)     # nums length
        
        # subarray of size m+1 is to be checked
        for i in range(n - m):
            valid = True
            # for each element in the pattern, check condition between consecutive numbers
            for k in range(m):
                if pattern[k] == 1:
                    if not (nums[i + k + 1] > nums[i + k]):
                        valid = False
                        break
                elif pattern[k] == 0:
                    if not (nums[i + k + 1] == nums[i + k]):
                        valid = False
                        break
                elif pattern[k] == -1:
                    if not (nums[i + k + 1] < nums[i + k]):
                        valid = False
                        break
            if valid:
                count += 1
        return count