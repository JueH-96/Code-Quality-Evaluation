class Solution:
    def calculateScore(self, s: str) -> int:
        # stacks[c] will hold indices of unmarked occurrences of letter 'a'+c
        stacks = [[] for _ in range(26)]
        score = 0
        for i, ch in enumerate(s):
            c = ord(ch) - ord('a')
            # mirror index in 0..25
            m = 25 - c
            if stacks[m]:
                # match with the closest unmarked mirror letter
                j = stacks[m].pop()
                score += i - j
            else:
                # no match, keep this index as unmarked for future
                stacks[c].append(i)
        return score