class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        from collections import Counter
        
        # Count frequency of all characters across all words
        char_count = Counter()
        for word in words:
            char_count.update(word)
        
        # Calculate available pairs and singles  
        pairs = sum(count // 2 for count in char_count.values())
        singles = sum(count % 2 for count in char_count.values())
        
        # Sort word lengths (greedy: try shorter words first)
        lengths = sorted(len(word) for word in words)
        
        result = 0
        for length in lengths:
            if length % 2 == 0:
                # Even length word needs length/2 pairs
                needed_pairs = length // 2
                if pairs >= needed_pairs:
                    pairs -= needed_pairs
                    result += 1
            else:
                # Odd length word needs (length-1)/2 pairs + 1 single
                needed_pairs = (length - 1) // 2
                if pairs >= needed_pairs:
                    if singles >= 1:
                        # Use existing single
                        pairs -= needed_pairs
                        singles -= 1
                        result += 1
                    elif pairs >= needed_pairs + 1:
                        # Break extra pair to get single
                        pairs -= (needed_pairs + 1)
                        singles += 1
                        result += 1
        
        return result