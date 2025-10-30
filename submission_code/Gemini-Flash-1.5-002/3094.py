class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        twos = 0
        threes = 0
        ones = 0
        
        for count in counts.values():
            if count >= 3:
                threes += count // 3
                count %= 3
            if count >= 2:
                twos += count // 2
                count %= 2
            ones += count

        if ones > twos + threes:
            return -1
        
        operations = threes
        ones -= threes * 3
        twos -= threes * 1
        
        operations += twos
        ones -= twos * 2
        
        operations += (ones + 2) // 3
        
        return operations