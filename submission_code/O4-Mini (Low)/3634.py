class Solution:
    def calculateScore(self, s: str) -> int:
        # stacks for unpaired indices of each character
        stacks = [[] for _ in range(26)]
        score = 0
        
        def mirror_idx(c_idx: int) -> int:
            # mirror of index i is (25 - i)
            return 25 - c_idx
        
        for i, ch in enumerate(s):
            c_idx = ord(ch) - ord('a')
            m_idx = mirror_idx(c_idx)
            # if there's an unmarked mirror character before i, pair it
            if stacks[m_idx]:
                j = stacks[m_idx].pop()
                score += i - j
            else:
                # otherwise keep this character as unmarked
                stacks[c_idx].append(i)
        
        return score