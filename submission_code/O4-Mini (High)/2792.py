from typing import List

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        """
        It's necessary and sufficient that the XOR of all derived values is 0.
        This is because summing (XOR-ing) all equations derived[i] = orig[i] ⊕ orig[i+1]
        around the cycle makes every orig[k] appear exactly twice, forcing
        XOR_all = 0 for consistency.
        """
        xor_sum = 0
        for bit in derived:
            xor_sum ^= bit
        return xor_sum == 0