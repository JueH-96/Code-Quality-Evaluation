class Solution:
    def finalString(self, s: str) -> str:
        result = []
        for ch in s:
            if ch != 'i':
                result.append(ch)
            else:
                result.reverse()  # Reverse the list when 'i' is encountered
        return ''.join(result)