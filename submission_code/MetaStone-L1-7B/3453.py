class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 0:
            return []
        current = ['0', '1']
        if n == 1:
            return current
        for _ in range(1, n):
            next_level = []
            for s in current:
                last = s[-1]
                if last == '0':
                    next_level.append(s + '1')
                else:
                    next_level.append(s + '0')
                    next_level.append(s + '1')
            current = next_level
        return current