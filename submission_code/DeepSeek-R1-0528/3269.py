class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        arr = []
        for i in range(n - 1):
            if nums[i+1] > nums[i]:
                arr.append(1)
            elif nums[i+1] == nums[i]:
                arr.append(0)
            else:
                arr.append(-1)
        
        m = len(pattern)
        count = 0
        for i in range(len(arr) - m + 1):
            match = True
            for j in range(m):
                if arr[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                count += 1
        return count