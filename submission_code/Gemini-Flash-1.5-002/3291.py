class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        count = {}
        for num in nums:
            bits = bin(num).count('1')
            if bits not in count:
                count[bits] = []
            count[bits].append(num)
        
        sorted_nums = []
        for bits in sorted(count.keys()):
            sorted_nums.extend(sorted(count[bits]))
        
        return sorted_nums == sorted(nums)