class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        count = 0
        for i in range(n - m):
            match = True
            for k in range(m):
                a = nums[i + k]
                b = nums[i + k + 1]
                p = pattern[k]
                if p == 1:
                    if not (b > a):
                        match = False
                        break
                elif p == 0:
                    if not (b == a):
                        match = False
                        break
                elif p == -1:
                    if not (b < a):
                        match = False
                        break
            if match:
                count += 1
        return count