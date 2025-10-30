class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # The idea: In each operation, we subtract (2^i + num2) from num1.
        # If we perform k operations with chosen i's (they can repeat),
        # the total subtraction is (sum(2^i) + k * num2). Let S = sum(2^i) (all terms are positive powers of 2).
        # We need num1 - k*num2 to equal S.
        # And note S, being a sum of k powers of 2 (each at least 1), must satisfy:
        #    S >= k   (each term is at least 1)
        # Also, representing S in binary, the count of ones of S, i.e. popcount(S),
        # is exactly the minimal number of powers of 2 needed to form S.
        # So a necessary and sufficient condition for S to be representable as exactly k powers of 2
        # (by possibly splitting a power-of-2 into smaller ones) is:
        #    popcount(S) <= k <= S
        # Here, S = num1 - k*num2.
        #
        # One more detail: we can only subtract powers of 2 up to 2^60 (i in [0, 60]).
        # Hence the maximum value we can get from one operation is (2^60).
        # So in k operations the maximum sum we can get is k * 2^60.
        # Thus we also need S <= k * (2^60).
        #
        # Our approach is: for increasing number of operations k (starting from 1),
        # let count_sum = num1 - k*num2 and check if it is nonnegative and satisfies:
        #   (a) count_sum >= k,  (because we need at least k when using k ones)
        #   (b) popcount(count_sum) <= k <= count_sum, and
        #   (c) count_sum <= k * (2^60).
        #
        # If num2 > 0 then num1 - k*num2 is decreasing in k.
        # So if for some k we have num1 - k*num2 < k, then for any larger k it will not be possible.
        #
        # We iterate for k from 1 to a safe upper bound. 
        # Based on the constraints, iterating up to 10^6 is safe.
        
        # The safe upper bound for k depends on num2.
        # If num2 > 0, T = num1 - k*num2 is decreasing, so we can break early.
        # If num2 <= 0, T increases with k, but condition (c) T <= k*(2**60) usually holds for k
        # because even in worst-case T = num1 + |num2| * k, so k needed is around num1/(2**60 - |num2|) 
        # which, given the constraints, will be very small relative to 10^6.
        
        # We use an iteration limit of 10^6.
        LIMIT = 10**6
        for k in range(1, LIMIT + 1):
            # T is the total sum of powers 2^i to be chosen
            T = num1 - k * num2
            
            # T must be non-negative to be represented as a sum of powers of 2.
            if T < 0:
                # If num2 > 0, then T will only decrease further.
                if num2 > 0:
                    break
                continue
            
            # The sum of k operations (each at least 1) must be at least k.
            if T < k:
                # For num2 > 0, no further k will make T >= k; break early.
                if num2 > 0:
                    break
                continue
            
            # Also, each subtraction can contribute at most 2^60.
            if T > k * (1 << 60):
                continue
            
            # Count the number of ones in T (which is the minimal number of powers of 2 needed).
            ones = bin(T).count("1")
            # We need to be able to split T into exactly k terms (you can always split a power of 2 into smaller ones)
            # if ones <= k <= T.
            if ones <= k <= T:
                return k
        return -1