class Solution:
    def minChanges(self, n: int, k: int) -> int:
        """
        We can only switch bits from 1 to 0 in n to match k.
        So, k must be a submask of n (i.e., all 1 bits in k
        are already 1 in n). Otherwise, it's impossible (-1).
        
        Steps:
        1. Check feasibility: if k has any '1' bit that n does not have,
           then it's impossible. In other words, (k & ~n) != 0 => -1.
        2. If feasible, the number of changes is the count of bits
           that are 1 in n and 0 in k. In other words, the number of set bits in (n & ~k).
        """
        # Check feasibility
        if (k & ~n) != 0:  # k has bits that n doesn't
            return -1
        
        # Count bits that are 1 in n but 0 in k
        diff_bits = n & ~k
        # count the set bits
        return bin(diff_bits).count('1')