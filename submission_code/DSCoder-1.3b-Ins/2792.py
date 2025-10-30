class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        original = derived.copy()
        for i in range(len(derived)):
            if i == len(derived) - 1:
                derived[i] = original[0] ^ original[i]
            else:
                derived[i] = original[i] ^ original[i + 1]
        return derived == original