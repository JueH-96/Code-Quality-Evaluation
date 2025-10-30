class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
        
        xor_result = 0
        for num, cnt in count.items():
            if cnt == 2:
                xor_result ^= num
        
        return xor_result