class Solution:
    def calculateScore(self, s: str) -> int:
        # Create a dictionary to hold stacks of unmarked indices for each letter.
        stacks = {chr(c): [] for c in range(ord('a'), ord('z') + 1)}
        score = 0
        
        # Process each character from left to right.
        for i, ch in enumerate(s):
            # Compute the mirror letter.
            mirror_ch = chr(ord('z') + ord('a') - ord(ch))
            # If there is an unmarked index with the mirror letter, pair and update the score.
            if stacks[mirror_ch]:
                j = stacks[mirror_ch].pop()  # Closest unmarked index for mirror letter
                score += (i - j)
            else:
                # Otherwise, mark this index as available for future pairing.
                stacks[ch].append(i)
                
        return score