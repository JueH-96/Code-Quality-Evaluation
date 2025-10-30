class Solution:
    def minLength(self, s: str) -> int:
        res = []
        for char in s:
            res.append(char)
            while len(res) >= 2:
                last_two = "".join(res[-2:])
                if last_two == "AB" or last_two == "CD":
                    res.pop()
                    res.pop()
                else:
                    break
        return len(res)