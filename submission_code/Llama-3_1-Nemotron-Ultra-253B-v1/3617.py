class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 1
        groups = []
        current = word[0]
        count = 1
        for c in word[1:]:
            if c == current:
                count += 1
            else:
                groups.append(count)
                current = c
                count = 1
        groups.append(count)
        total = sum(g - 1 for g in groups)
        return total + 1