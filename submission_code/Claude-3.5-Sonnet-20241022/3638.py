class Solution:
    def makeStringGood(self, s: str) -> int:
        # Count frequency of each character
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        
        # Get all unique frequencies
        frequencies = list(freq.values())
        
        # If all frequencies are already same, return 0
        if len(set(frequencies)) == 1:
            return 0
        
        min_ops = float('inf')
        # Try to make all characters have frequency f
        for target in range(1, max(frequencies) + 2):
            curr_ops = 0
            
            # For each character frequency
            for f in frequencies:
                if f < target:
                    # Need to add or change characters
                    curr_ops += target - f
                elif f > target:
                    # Need to delete characters
                    curr_ops += f - target
            
            min_ops = min(min_ops, curr_ops)
        
        return min_ops