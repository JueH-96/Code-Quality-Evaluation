import math

class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        """
        Finds the minimum number of operations k to make num1 zero.
        In one operation, subtract 2^i + num2 from num1, where 0 <= i <= 60.

        Args:
            num1: The starting integer (1 <= num1 <= 10^9).
            num2: The integer subtracted along with 2^i (-10^9 <= num2 <= 10^9).

        Returns:
            The minimum number of operations k, or -1 if impossible.
        """
        
        # Let k be the number of operations.
        # After k operations, the value of num1 becomes:
        # num1' = num1 - sum_{j=1}^{k} (2^{i_j} + num2)  where 0 <= i_j <= 60
        # We want the final value num1' to be 0.
        # Setting num1' = 0:
        # num1 - (sum_{j=1}^{k} 2^{i_j}) - k * num2 = 0
        # num1 - k * num2 = sum_{j=1}^{k} 2^{i_j}
        
        # Let target = num1 - k * num2.
        # The problem reduces to finding the minimum integer k >= 1 such that 'target' 
        # can be expressed as the sum of exactly k powers of 2, where each power's 
        # exponent i_j is in the range [0, 60].
        # That is, target = 2^{i_1} + 2^{i_2} + ... + 2^{i_k}, with 0 <= i_j <= 60.

        # A non-negative integer x can be represented as the sum of exactly k powers of 2 
        # if and only if two conditions are met:
        # 1. The number of set bits (popcount) in the binary representation of x is less than or equal to k. 
        #    popcount(x) <= k
        #    Reason: popcount(x) is the minimum number of powers of 2 required if they must be distinct. 
        #    Allowing repeated powers (e.g., replacing 2^p with 2^(p-1) + 2^(p-1)) can increase the number of terms.
        # 2. The value x itself must be greater than or equal to k.
        #    x >= k
        #    Reason: The smallest possible sum of k powers of 2 is k * 2^0 = k.
        
        # Additionally, the powers used must have exponents i_j <= 60.
        # Based on the constraints (num1 <= 10^9, |num2| <= 10^9) and iterating k up to around 61, 
        # the maximum value of target = num1 - k * num2 remains relatively small (around 65 * 10^9, which is < 2^37). 
        # Any representation of such a target as a sum of powers of 2 will involve exponents much smaller than 60.
        # Therefore, the constraint i_j <= 60 is implicitly satisfied if the conditions popcount(target) <= k and target >= k hold
        # for k in a reasonable range (like 1 to 61).

        # We iterate through possible values of k, starting from 1, and check if the conditions are met.
        # The number of possible choices for 'i' in the operation is 61 (from 0 to 60). This suggests
        # that the number of operations k might not need to exceed 61. We iterate k from 1 to 61.
        
        for k in range(1, 62): # Check k from 1 up to 61
            target = num1 - k * num2

            # Check if target can be represented as a sum of k powers of 2.
            # Condition 1: target >= k
            # Also, target must be non-negative to be a sum of powers of 2.
            # target >= k implies target >= 1 (since k >= 1), so non-negativity is covered.
            if target < k:
                # If target < k, this value of k is invalid.
                
                # Optimization: If num2 >= 0, then target = num1 - k * num2 is non-increasing as k increases.
                # Since k increases, the difference target - k will only decrease further.
                # If target < k holds once, it will hold for all larger k.
                # Thus, if num2 >= 0, we can conclude that no solution exists.
                if num2 >= 0:
                    return -1
                
                # If num2 < 0, target = num1 - k * num2 increases as k increases.
                # It's possible that target >= k might be satisfied for a larger k.
                # So, we just continue to the next value of k.
                else:
                    continue 

            # At this point, we have target >= k.
            
            # Check Condition 2: popcount(target) <= k
            # Calculate the number of set bits (1s) in the binary representation of target.
            # Python 3.10+ has target.bit_count(). Using bin().count() for broader compatibility.
            popcount = bin(target).count('1') 

            if popcount <= k:
                # Both necessary conditions are met: target >= k and popcount(target) <= k.
                # Our analysis suggests this is sufficient under the given constraints.
                # Since we are iterating k from 1 upwards, this is the minimum k that works.
                return k

        # If the loop completes without finding a suitable k in the range [1, 61],
        # it means it's impossible to make num1 zero under the given rules.
        return -1