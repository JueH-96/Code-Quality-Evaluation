class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def original_partitions(s, k_val):
            n = len(s)
            partitions = 0
            start = 0
            while start < n:
                partitions += 1
                current_chars = set()
                end = start
                while end < n:
                    if s[end] in current_chars:
                        end += 1
                    else:
                        if len(current_chars) < k_val:
                            current_chars.add(s[end])
                            end += 1
                        else:
                            break
                start = end
            return partitions
        
        original = original_partitions(s, k)
        max_partitions = original
        n = len(s)
        
        # Iterate over each possible position to change
        for i in range(n):
            # Try changing to every possible character
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if s[i] == c:
                    continue
                # Create modified string
                modified = s[:i] + c + s[i+1:]
                current = original_partitions(modified, k)
                if current > max_partitions:
                    max_partitions = current
        return max_partitions