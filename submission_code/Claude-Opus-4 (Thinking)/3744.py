class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        def min_operations_for_query(l, r):
            total_work = 0
            k = 1
            
            while True:
                if k == 1:
                    lower = 1
                    upper = 3
                else:
                    lower = 4 ** (k - 1)
                    upper = 4 ** k - 1
                
                if lower > r:
                    break
                
                # Count how many numbers in [l, r] fall in [lower, upper]
                count = max(0, min(r, upper) - max(l, lower) + 1)
                # Each of these numbers needs k operations
                total_work += k * count
                k += 1
            
            # Each operation processes 2 units of work
            return (total_work + 1) // 2  # ceil(total_work / 2)
        
        result = 0
        for l, r in queries:
            result += min_operations_for_query(l, r)
        return result