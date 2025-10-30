class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = [0] * 51
        
        for start in range(n - k + 1):
            seen = set()
            for i in range(start, start + k):
                seen.add(nums[i])
            for num in seen:
                count[num] += 1
        
        for x in range(50, -1, -1):
            if count[x] == 1:
                return x
        
        return -1