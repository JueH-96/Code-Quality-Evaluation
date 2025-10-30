class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        for original_0 in [0, 1]:
            original = [0] * n
            original[0] = original_0
            valid = True
            for i in range(n):
                next_index = (i + 1) % n
                next_value = (derived[i] + original[i]) % 2
                if next_index == 0:
                    if next_value != original[0]:
                        valid = False
                        break
                else:
                    original[next_index] = next_value
            if valid:
                return True
        return False