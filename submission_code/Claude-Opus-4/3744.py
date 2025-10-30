class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        def min_ops_for_range(l, r):
            # Track frequency of each number
            freq = {}
            for num in range(l, r + 1):
                freq[num] = freq.get(num, 0) + 1
            
            operations = 0
            
            while any(num > 0 for num in freq):
                new_freq = {}
                total_count = sum(count for num, count in freq.items() if num > 0)
                
                if total_count == 0:
                    break
                
                # Number of operations needed this round
                ops_needed = (total_count + 1) // 2
                operations += ops_needed
                
                # Apply operations
                for num, count in freq.items():
                    if num > 0:
                        new_num = num // 4
                        if new_num > 0:
                            new_freq[new_num] = new_freq.get(new_num, 0) + count
                
                freq = new_freq
            
            return operations
        
        result = 0
        for query in queries:
            result += min_ops_for_range(query[0], query[1])
        
        return result