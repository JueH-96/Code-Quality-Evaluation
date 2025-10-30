class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        # Try both possibilities for original[0]
        for first in [0, 1]:
            original = [first]
            for i in range(n - 1):
                next_bit = original[i] ^ derived[i]
                original.append(next_bit)
            # Check if the last derived condition holds
            if (original[-1] ^ original[0]) == derived[-1]:
                return True
        return False