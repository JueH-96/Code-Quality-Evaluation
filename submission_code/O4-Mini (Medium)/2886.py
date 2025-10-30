class Solution:
    def finalString(self, s: str) -> str:
        res = []
        for c in s:
            if c == 'i':
                # Faulty key: reverse the current text
                res.reverse()
            else:
                # Normal key: append the character
                res.append(c)
        return "".join(res)