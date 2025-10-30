class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # The XOR of all elements in the derived array should be 0 for a valid original array to exist
        xor_sum = 0
        for num in derived:
            xor_sum ^= num
        return xor_sum == 0