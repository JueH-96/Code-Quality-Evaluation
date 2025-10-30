class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        counts = {}
        duplicates = []
        for num in nums:
            if num in counts:
                counts[num] += 1
            else:
                counts[num] = 1
        
        for num, count in counts.items():
            if count == 2:
                duplicates.append(num)
        
        result = 0
        for num in duplicates:
            result ^= num
            
        return result