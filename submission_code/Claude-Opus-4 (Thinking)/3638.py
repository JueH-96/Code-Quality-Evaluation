class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        
        freq = Counter(s)
        n = len(s)
        
        min_ops = n  # Worst case: remove all characters
        
        # Try each possible target frequency
        for target in range(1, n + 1):
            # Get all unique characters sorted
            chars = sorted(freq.keys())
            
            # Calculate excess/deficit for each character
            balance = {}
            for c in chars:
                balance[c] = freq[c] - target
            
            # Use "change" operations to transfer between consecutive characters
            operations = 0
            
            for i in range(len(chars) - 1):
                c1, c2 = chars[i], chars[i + 1]
                if ord(c2) == ord(c1) + 1:  # Consecutive in alphabet
                    if balance[c1] > 0 and balance[c2] < 0:
                        # Transfer excess from c1 to deficit in c2
                        transfer = min(balance[c1], -balance[c2])
                        operations += transfer
                        balance[c1] -= transfer
                        balance[c2] += transfer
            
            # Remaining imbalances need delete (if positive) or insert (if negative)
            for c in chars:
                operations += abs(balance[c])
            
            min_ops = min(min_ops, operations)
        
        return min_ops