class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        bits = []
        for num in nums:
            count = 0
            while num > 0:
                if num & 1:
                    count += 1
                num >>= 1
            bits.append(count)
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] > nums[j] and bits[i] != bits[j]:
                    return False
        return True