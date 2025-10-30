class Solution:
    def smallestNumber(self, n: int) -> int:
        
        # The numbers whose binary representation consists only of set bits
        # are of the form 2^k - 1. For k=1, 2, 3, ... these numbers are
        # 1, 3, 7, 15, 31, and so on. This is a strictly increasing sequence.
        
        # We are looking for the smallest number in this sequence that is
        # greater than or equal to n.
        
        # We can generate these numbers iteratively, starting from the smallest,
        # until we find one that meets the condition.
        
        # Start with the first number in the sequence, which is 1.
        candidate = 1
        
        # The next number in the sequence can be generated from the previous one
        # by a left shift and setting the new least significant bit.
        # For example, from 7 (binary '111'), we get (7 << 1) | 1 = 15 (binary '1111').
        while candidate < n:
            candidate = (candidate << 1) | 1
            
        # The loop stops when `candidate` is greater than or equal to `n`.
        # Since we generate the candidates in increasing order, this is the
        # smallest such number.
        return candidate