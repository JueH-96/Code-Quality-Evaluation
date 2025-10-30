class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        result = 0
        
        for l, r in queries:
            total_operations = 0
            
            # Find k_n values for l and r
            k_l = (l.bit_length() + 1) // 2
            k_r = (r.bit_length() + 1) // 2
            
            for k_value in range(k_l, k_r + 1):
                # Compute the range for this k_value
                if k_value == 1:
                    start = 1
                else:
                    start = 1 << (2 * k_value - 2)  # 2^(2*k_value-2)
                end = (1 << (2 * k_value)) - 1  # 2^(2*k_value)-1
                
                # Adjust the range for the query boundaries
                start = max(start, l)
                end = min(end, r)
                
                # Compute the contribution of this segment
                count = end - start + 1
                if count > 0:
                    total_operations += k_value * count
            
            result += (total_operations + 1) // 2
        
        return result