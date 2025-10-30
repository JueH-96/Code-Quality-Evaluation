class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        return derived[0] == derived[-1]