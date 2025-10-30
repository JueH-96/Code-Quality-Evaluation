class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s):
            partitions = 0
            i = 0
            n = len(s)
            
            while i < n:
                distinct = set()
                j = i
                
                while j < n:
                    if s[j] not in distinct:
                        if len(distinct) == k:
                            break
                        distinct.add(s[j])
                    j += 1
                
                partitions += 1
                i = j
            
            return partitions
        
        # Get baseline partitions without any changes
        max_partitions = count_partitions(s)
        
        # Try changing each character
        s_list = list(s)
        n = len(s)
        
        # Get all characters that appear in the string
        all_chars = set(s)
        
        for i in range(n):
            original_char = s_list[i]
            
            # Try changing to characters that appear in the string + a few others
            candidates = all_chars.copy()
            candidates.add('a')  # Always try 'a' as it's common
            
            for new_char in candidates:
                if new_char != original_char:
                    s_list[i] = new_char
                    partitions = count_partitions(''.join(s_list))
                    max_partitions = max(max_partitions, partitions)
            
            # Restore original character
            s_list[i] = original_char
        
        return max_partitions