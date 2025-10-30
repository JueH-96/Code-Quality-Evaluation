class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        
        # Count the frequency of each character
        char_count = Counter(s)
        
        # Find the most common frequency
        most_common_freq = max(char_count.values())
        
        # Calculate the total operations needed
        total_ops = 0
        for char, count in char_count.items():
            if char != 'z':
                # For characters other than 'z', we can change them to the next letter
                # or delete/insert to match the most common frequency
                total_ops += min(most_common_freq - count, count)
            else:
                # For 'z', we can only delete or insert
                total_ops += abs(most_common_freq - count)
        
        return total_ops