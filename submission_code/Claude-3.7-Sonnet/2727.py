class Solution:
    def countSeniors(self, details: List[str]) -> int:
        count = 0
        for detail in details:
            age = int(detail[11:13])  # Extract age from positions 11-12
            if age > 60:
                count += 1
        return count