class Solution:
    def countSeniors(self, details: List[str]) -> int:
        result = 0
        for entry in details:
            # Age is located at indices 11 and 12
            age = int(entry[11:13])
            if age > 60:
                result += 1
        return result