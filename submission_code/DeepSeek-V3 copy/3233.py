class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        max_partitions = 0
        
        # Function to calculate the number of partitions without any changes
        def calculate_partitions(s):
            partitions = 0
            i = 0
            while i < n:
                unique = set()
                j = i
                while j < n and len(unique) <= k:
                    unique.add(s[j])
                    if len(unique) > k:
                        break
                    j += 1
                partitions += 1
                i = j
            return partitions
        
        # Initial partitions without any changes
        initial_partitions = calculate_partitions(s)
        max_partitions = initial_partitions
        
        # Try changing each character to every possible lowercase letter
        for i in range(n):
            original_char = s[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == original_char:
                    continue
                # Create a new string with the character changed
                new_s = s[:i] + c + s[i+1:]
                # Calculate the number of partitions for the new string
                current_partitions = calculate_partitions(new_s)
                if current_partitions > max_partitions:
                    max_partitions = current_partitions
            # Restore the original character
            s = s[:i] + original_char + s[i+1:]
        
        return max_partitions