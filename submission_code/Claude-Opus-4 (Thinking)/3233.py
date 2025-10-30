class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(string, k, change_idx=-1, new_char=''):
            partitions = 0
            i = 0
            n = len(string)
            
            while i < n:
                distinct_chars = set()
                
                # Find longest prefix starting at i with at most k distinct chars
                for j in range(i, n):
                    # Use the changed character if we're at the change index
                    char = new_char if j == change_idx else string[j]
                    distinct_chars.add(char)
                    
                    # If we exceed k distinct chars, end the current partition
                    if len(distinct_chars) > k:
                        partitions += 1
                        i = j
                        break
                else:
                    # We reached the end without exceeding k distinct chars
                    partitions += 1
                    i = n
            
            return partitions
        
        # Count partitions without any change
        max_partitions = count_partitions(s, k)
        
        # Try changing each position to each character
        n = len(s)
        for i in range(n):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != s[i]:
                    partitions = count_partitions(s, k, i, c)
                    max_partitions = max(max_partitions, partitions)
        
        return max_partitions