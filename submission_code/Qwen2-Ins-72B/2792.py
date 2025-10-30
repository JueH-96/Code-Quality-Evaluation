class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        for original_0 in [0, 1]:
            original = [original_0]
            for i in range(n - 1):
                original.append(derived[i] ^ original[-1])
            if derived[-1] == original[-1] ^ original[0]:
                return True
        return False