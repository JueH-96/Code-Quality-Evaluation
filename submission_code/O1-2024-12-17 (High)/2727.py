class Solution:
    def countSeniors(self, details: List[str]) -> int:
        count = 0
        for info in details:
            age = int(info[11:13])
            if age > 60:
                count += 1
        return count