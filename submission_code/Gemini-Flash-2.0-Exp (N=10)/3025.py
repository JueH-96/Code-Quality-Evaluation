class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        operations = 0
        
        for bit in range(31):
            if (target >> bit) & 1:
                if (1 << bit) in counts and counts[1 << bit] > 0:
                    counts[1 << bit] -= 1
                else:
                    found = False
                    for j in range(bit + 1, 31):
                        if (1 << j) in counts and counts[1 << j] > 0:
                            counts[1 << j] -= 1
                            for k in range(j - 1, bit - 1, -1):
                                counts[1 << k] = counts.get(1 << k, 0) + 2
                                operations += 1
                            found = True
                            break
                    if not found:
                        return -1
        return operations