class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        # Try starting with original[0] = 0
        original = [0] * n
        original[0] = 0
        for i in range(n-1):
            original[i+1] = original[i] ^ derived[i]
        # Check if the last derived condition holds
        if (original[-1] ^ original[0]) == derived[-1]:
            return True
        # Try starting with original[0] = 1
        original = [0] * n
        original[0] = 1
        for i in range(n-1):
            original[i+1] = original[i] ^ derived[i]
        # Check if the last derived condition holds
        if (original[-1] ^ original[0]) == derived[-1]:
            return True
        return False