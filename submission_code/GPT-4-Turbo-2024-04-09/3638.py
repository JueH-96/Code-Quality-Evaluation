class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        
        # Count frequency of each character
        freq = Counter(s)
        
        # Get the frequencies of characters and sort them
        freq_values = sorted(freq.values())
        
        # If all frequencies are the same already, no operation is needed
        if len(set(freq_values)) == 1:
            return 0
        
        # Calculate minimum operations required to make all frequencies equal
        min_operations = float('inf')
        
        # Try to make all characters occur `target` times
        for target in range(1, max(freq_values) + 1):
            current_operations = 0
            
            for char, count in freq.items():
                if count > target:
                    # We need to remove some characters
                    current_operations += (count - target)
                elif count < target:
                    # We need to add some characters
                    current_operations += (target - count)
            
            # Update minimum operations if the current strategy uses fewer operations
            min_operations = min(min_operations, current_operations)
        
        return min_operations