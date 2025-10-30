class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        def count_operations(n):
            if n == 0:
                return 0
            ops = 0
            while n > 0:
                n //= 4
                ops += 1
            return ops
        
        result = 0
        for l, r in queries:
            total_individual_ops = 0
            for num in range(l, r + 1):
                total_individual_ops += count_operations(num)
            # Since each operation processes 2 numbers, we need ceil(total/2) operations
            result += (total_individual_ops + 1) // 2
        
        return result