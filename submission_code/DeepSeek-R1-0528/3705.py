class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n = len(nums)
        freq = [0] * 51
        
        for i in range(n - k + 1):
            distinct = set(nums[i:i+k])
            for num in distinct:
                freq[num] += 1
        
        for num in range(50, -1, -1):
            if freq[num] == 1:
                return num
        return -1