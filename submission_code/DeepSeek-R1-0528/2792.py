class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        total = 0
        for d in derived:
            total ^= d
        return total == 0