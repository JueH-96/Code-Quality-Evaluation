class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        groups = [[] for _ in range(k)]
        
        # Group characters based on their position modulo k
        for i in range(n):
            groups[i % k].append(word[i])
        
        total_operations = 0
        
        for group in groups:
            char_count = {}
            group_size = len(group)
            
            # Count occurrences of each character in the group
            for char in group:
                char_count[char] = char_count.get(char, 0) + 1
            
            # Find the most frequent character
            max_count = max(char_count.values())
            
            # Calculate operations needed for this group
            operations = group_size - max_count
            total_operations += operations
        
        return total_operations