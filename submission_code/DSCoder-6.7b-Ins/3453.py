class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        else:
            result = []
            for s in self.validStrings(n-1):
                result.append(s + "0")
                result.append(s + "1")
                if s[-1] == "0":
                    result.append(s[:-1] + "1")
            return result