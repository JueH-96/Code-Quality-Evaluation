class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        m = len(pattern)
        n = len(nums)
        count = 0
        
        for i in range(n - m):
            match = True
            for k in range(m):
                a = nums[i + k]
                b = nums[i + k + 1]
                if pattern[k] == 1:
                    if b <= a:
                        match = False
                        break
                elif pattern[k] == 0:
                    if a != b:
                        match = False
                        break
                elif pattern[k] == -1:
                    if b >= a:
                        match = False
                        break
            if match:
                count += 1
        return count