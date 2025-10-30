class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # The XOR of all elements in the derived array should be 0 for a valid original array to exist.
        # This is because the XOR of all elements in the derived array is equivalent to the XOR of all elements in the original array.
        # If the XOR of all elements in the derived array is 0, then there exists a valid original array.
        # Otherwise, there does not exist a valid original array.
        return all(x == 0 for x in derived)