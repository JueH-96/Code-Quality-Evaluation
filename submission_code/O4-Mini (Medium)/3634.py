class Solution:
    def calculateScore(self, s: str) -> int:
        # stacks[c] will hold the indices of unmarked occurrences of the character
        # with alphabetical index c (0 for 'a', ..., 25 for 'z')
        stacks = [[] for _ in range(26)]
        score = 0
        
        for i, ch in enumerate(s):
            idx = ord(ch) - ord('a')
            mirror_idx = 25 - idx  # mirror letter's index in [0..25]
            
            # If there's an unmarked mirror character before i, match it
            if stacks[mirror_idx]:
                j = stacks[mirror_idx].pop()
                score += i - j
            else:
                # Otherwise, record this index as unmarked for future matches
                stacks[idx].append(i)
        
        return score