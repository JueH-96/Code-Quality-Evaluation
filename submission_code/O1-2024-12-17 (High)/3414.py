class Solution:
    def waysToReachStair(self, k: int) -> int:
        """
        We want to count all distinct sequences of operations taking Alice from stair 1 to stair k.
        
        Operations:
          1) Down: from i -> i - 1 (cannot be done consecutively, and cannot be done from i=0)
          2) Up:   from i -> i + 2^jump, then jump += 1

        Key insight:
        - Each time we use an Up operation, the amount we go up is 2^(current_jump), 
          where current_jump increases from 0 to 1, 1 to 2, etc., in that fixed ascending order.
        - If we do n Up operations total, then the total "up" amount is 2^0 + 2^1 + ... + 2^(n - 1) = 2^n - 1.
        - Starting at stair 1, if we use n Ups (summing to 2^n - 1) and d Downs (each Down = 1 step), 
          the final position will be 1 + (2^n - 1) - d = 2^n - d.

        To reach stair k, we require 2^n - d = k. Hence d = 2^n - k.

        Constraints to respect:
        - d must be ≥ 0 (so 2^n >= k).
        - We cannot do two Downs in a row. Equivalently, in the sequence of n Ups (fixed order) plus d Downs, 
          no two Downs can appear consecutively.
        - We also cannot apply a Down when currently at stair 0, but this is automatically avoided 
          provided no two Downs are consecutive (starting from 1, each Down can only bring us to 0 but 
          the next step must be an Up, not another Down).

        Counting method:
        - If we fix n (the total number of Ups), we have (n+1) "slots" around those Ups where Downs could be placed:
            S0 (before the 1st Up), S1 (between Up(0) and Up(1)), ..., S(n-1), Sn (after the nth Up).
        - Because no two Downs can be consecutive, we can place at most one Down in each slot.
        - We need exactly d = 2^n - k Downs in total. So the number of ways, for a fixed n, 
          is "choose(n+1, d)" if 0 <= d <= n+1.
        - We sum over all n such that 2^n >= k (so d >= 0). For large n, if d > n+1, that term is 0.

        Hence the final formula for ways(k) is:
            ways(k) = sum_{n=0,1,2,...} of C(n+1, 2^n - k),
            counting only terms where 0 <= 2^n - k <= n+1.

        Because k <= 10^9, we only need to check up to n ≈ 30–32 (since 2^30 > 10^9).

        Examples check:
        - k=0 => answer=2
        - k=1 => answer=4
        - k=2 => answer=4
        - k=3 => answer=3
        - k=4 => answer=2
        These match the reasoning and the binomial-sum formula.
        """

        import math

        ways = 0
        # 2^30 is a bit over 10^9, so going up to ~35 is plenty of margin
        for n in range(40):
            power = 1 << n  # 2^n
            if power < k:
                # If 2^n < k, then 2^n - k < 0 => no contribution
                continue
            
            diff = power - k  # This is d = number of Downs needed for n Ups
            if diff > n + 1:
                # If diff is bigger than the total slots n+1, no point continuing to larger n
                break

            # Add binomial coefficient C(n+1, diff) if valid
            # math.comb(a, b) = 0 automatically if b<0 or b>a in Python 3.8+,
            # but we'll rely on the conditions above to ensure it's valid.
            ways += math.comb(n + 1, diff)
        
        return ways