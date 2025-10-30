class Solution:
    def finalString(self, s: str) -> str:
        result = []
        for ch in s:
            if ch == 'i':
                result.reverse()
            else:
                result.append(ch)
        return "".join(result)