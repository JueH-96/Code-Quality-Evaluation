class Solution:
    def calculateScore(self, s: str) -> int:
        total_score = 0
        # Initialize a stack (list) for each character
        stacks = {c: [] for c in 'abcdefghijklmnopqrstuvwxyz'}
        
        for i, ch in enumerate(s):
            # Calculate the mirror character
            mirror_ch = chr(ord('z') - (ord(ch) - ord('a')))
            if stacks[mirror_ch]:
                # Found an unmarked index j where s[j] is the mirror of s[i]
                j = stacks[mirror_ch].pop()
                total_score += i - j
            else:
                # No such index j exists, so we add current index to the stack
                stacks[ch].append(i)
                
        return total_score