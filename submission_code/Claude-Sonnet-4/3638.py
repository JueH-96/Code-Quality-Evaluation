class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        import string
        
        freq = Counter(s)
        max_freq = max(freq.values()) if freq else 0
        min_ops = float('inf')
        
        # Try each possible target frequency
        for target in range(max_freq + 1):
            ops = 0
            # Create array for current frequencies
            current = [0] * 26
            
            # Initialize with actual frequencies
            for c, f in freq.items():
                current[ord(c) - ord('a')] = f
            
            # Process from right to left (z to a) to maximize changes
            for i in range(25, -1, -1):  # z to a
                if current[i] > target:
                    excess = current[i] - target
                    # Try to change to next character (i+1)
                    if i < 25:  # not 'z'
                        need_next = max(0, target - current[i + 1])
                        change = min(excess, need_next)
                        current[i + 1] += change
                        excess -= change
                    # Delete remaining excess
                    ops += excess
                    current[i] = target
            
            # Add insertions needed
            for i in range(26):
                if current[i] < target:
                    ops += target - current[i]
            
            min_ops = min(min_ops, ops)
        
        return min_ops