class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        original = [0]*n
        for i in range(n-1, -1, -1):
            original[i] = derived[i] ^ original[(i+1)%n]
        return original == derived[::-1]