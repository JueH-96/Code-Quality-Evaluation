class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 0:
            return []
        current = ["0", "1"]
        for i in range(2, n + 1):
            next_list = []
            for s in current:
                if s[-1] == '0':
                    next_list.append(s + '1')
                else:
                    next_list.append(s + '0')
                    next_list.append(s + '1')
            current = next_list
        return current