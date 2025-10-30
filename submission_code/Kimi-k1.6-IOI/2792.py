class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        for o0 in [0, 1]:
            current = o0
            for i in range(n - 1):
                current ^= derived[i]
            if (current ^ o0) == derived[-1]:
                return True
        return False