class Solution:
    def countSeniors(self, details: List[str]) -> int:
        return sum(int(a[11:13]) > 60 for a in details)