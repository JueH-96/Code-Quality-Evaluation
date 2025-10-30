class Solution:
    def countKeyChanges(self, s: str) -> int:
        converted = s.lower()
        count = 0
        for i in range(1, len(converted)):
            if converted[i] != converted[i-1]:
                count += 1
        return count