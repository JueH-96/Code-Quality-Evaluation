class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        from itertools import combinations
        
        all_parts = set()
        
        # Generate all possible ways to choose (numFriends-1) cutting positions
        for positions in combinations(range(1, len(word)), numFriends - 1):
            # Add start and end positions to the cutting positions
            split_positions = [0] + list(positions) + [len(word)]
            
            # Extract all parts based on these positions
            parts = [word[split_positions[i]:split_positions[i+1]] 
                    for i in range(numFriends)]
            
            # Add all parts to our collection
            all_parts.update(parts)
        
        # Return the lexicographically largest string
        return max(all_parts)