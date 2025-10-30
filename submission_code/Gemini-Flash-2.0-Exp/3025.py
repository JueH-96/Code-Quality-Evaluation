class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        nums.sort(reverse=True)
        operations = 0
        current_sum = 0
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        for i in range(31, -1, -1):
            val = 1 << i
            
            if target >= val:
                needed = target // val
                target %= val
                
                available = counts.get(val, 0)
                
                if available >= needed:
                    current_sum += needed * val
                    counts[val] -= needed
                else:
                    current_sum += available * val
                    
                    counts[val] = 0
                    
                    j = i + 1
                    while j <= 31 and counts.get(1 << j, 0) == 0:
                        j += 1
                        operations += 1
                    
                    if j > 31:
                        return -1
                    
                    counts[1 << j] -= 1
                    
                    for k in range(i, j):
                        counts[1 << k] = counts.get(1 << k, 0) + 1
        
        return operations