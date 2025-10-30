class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        if k == 0:
            return 0  # though k >=1 per constraints
        
        count = 0
        
        from collections import defaultdict
        
        left = 0
        freq = defaultdict(int)
        required = defaultdict(int)
        for i in range(n):
            char = word[i]
            freq[char] += 1
        
        # Initially, check if the entire word is a valid substring
        # but it's better to process step by step
        # Let's iterate over all possible starting points
        for start in range(n):
            # We'll track the end of the current valid window
            end = start
            valid = True
            char_count = defaultdict(int)
            for end in range(start, n):
                char = word[end]
                char_count[char] += 1
                
                # Check if any character exceeds k
                if char_count[char] > k:
                    valid = False
                    break
                
                # If all characters are within k, check adjacency condition
                if all(v == k for v in char_count.values()):
                    # Now check adjacency condition
                    adj_valid = True
                    for i in range(start, end):
                        if i == start:
                            prev_char = None
                        else:
                            prev_char = word[i-1]
                        current_char = word[i]
                        if abs(i - (i-1)) > 2:
                            adj_valid = False
                            break
                    if adj_valid:
                        count += 1
        
        return count