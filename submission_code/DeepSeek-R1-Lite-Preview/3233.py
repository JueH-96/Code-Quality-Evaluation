class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        
        # Function to compute number of partitions for a given string
        def compute_partitions(s, k):
            partitions = 0
            left = 0
            seen = {}
            for right in range(len(s)):
                if s[right] in seen:
                    seen[s[right]] += 1
                else:
                    seen[s[right]] = 1
                if len(seen) > k:
                    partitions += 1
                    # Reset the window starting from left
                    new_seen = {}
                    new_seen[s[right]] = 1
                    left = right
                    seen = new_seen
            # Add the last partition
            partitions += 1
            return partitions
        
        # Compute original partitions
        original_partitions = compute_partitions(s, k)
        
        # Try changing each character to any other lowercase English letter
        max_partitions = original_partitions
        for i in range(n):
            original_char = s[i]
            # Try changing to each possible character except the current one
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == original_char:
                    continue
                # Create the modified string
                modified_s = s[:i] + c + s[i+1:]
                # Compute partitions for the modified string
                partitions = compute_partitions(modified_s, k)
                if partitions > max_partitions:
                    max_partitions = partitions
                # Early exit if we can't get more partitions
                if max_partitions == n:
                    return max_partitions
        return max_partitions