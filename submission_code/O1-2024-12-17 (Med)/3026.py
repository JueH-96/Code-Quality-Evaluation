class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        """
        We want an array of length n, with all distinct positive integers, such that
        no pair of distinct elements sums to 'target'. We must return the minimal
        possible sum of such an array modulo 10^9+7.

        Key idea:
        - If we pick a number x < target, we must avoid picking (target - x).
        - If target is even, picking target//2 is okay (since we can't pick it twice anyway).
        - Therefore, from the range 1..(target-1), we can at most pick floor((target-1)//2) numbers
          plus at most one extra if target is even (namely target//2).
        - Past that point, all integers >= target do not conflict with any chosen number,
          so we can pick freely starting from 'target' upward to fill the remainder.

        Let M = (target - 1)//2  (the count of "low" integers we can safely pick)
        Let halfPick = 1 if target is even else 0   (whether we can also pick target//2)
        
        Then the maximum we can pick from [1..(target-1)] plus possibly target//2 is M + halfPick.

        We compare n with these counts:
          1) If n <= M, we just pick the first n integers [1..n].
          2) Else if n <= M + halfPick, we pick [1..M] plus target//2 if needed (exactly once).
          3) Otherwise, we pick all of those (1..M plus target//2 if even) and then pick
             the remainder from [target, target+1, ...] until we have n distinct values.
        
        We compute the sum in O(1) time via closed-form formulas:
          sum_1_to_k = k*(k+1)//2
          sum_of_consecutive(r terms starting at A) = r*(2A + r - 1)//2

        Everything is taken modulo 10^9+7.
        """
        mod = 10**9 + 7
        
        # Helper to compute sum(1..k) % mod safely:
        def sum_1_to_k(k: int) -> int:
            # sum(1..k) = k*(k+1)//2
            return (k % mod) * ((k+1) % mod) % mod * pow(2, mod-2, mod) % mod
        
        # Helper to compute sum of 'count' consecutive integers starting at 'start'
        # i.e. sum( start, start+1, ..., start+count-1 ).
        def sum_consecutive(start: int, count: int) -> int:
            # sum = count*(2*start + count - 1)/2
            # do everything modded
            if count <= 0:
                return 0
            start_mod = start % mod
            count_mod = count % mod
            two_start_plus_count_minus_1 = (2*start_mod + count_mod - 1) % mod
            return (count_mod * two_start_plus_count_minus_1) % mod * pow(2, mod-2, mod) % mod
        
        # Main logic:
        M = (target - 1) // 2           # number of small picks below target/2
        halfPick = 1 if (target % 2 == 0) else 0
        halfVal = target // 2 if halfPick == 1 else 0
        
        # Case 1: If n <= M, pick 1..n
        if n <= M:
            return sum_1_to_k(n) % mod
        
        # Case 2: If n <= M + halfPick, we pick 1..M plus (target//2) if needed
        if n <= M + halfPick:
            # sum of 1..M plus (n-M) copies of halfVal (which is at most 1 copy)
            # If halfPick=1 and n=M+1 then we add halfVal once
            return (sum_1_to_k(M) + (n - M) * halfVal) % mod
        
        # Case 3: We pick all of 1..M plus halfVal (if even), then pick the leftover from [target..]
        leftover = n - (M + halfPick)  # how many more we need beyond the M + (possible) halfVal
        # sum(1..M):
        part_small = sum_1_to_k(M)
        # sum of halfVal if halfPick=1:
        part_half = (halfVal if halfPick == 1 else 0)
        # now pick `leftover` numbers starting from 'target'
        part_large = sum_consecutive(target, leftover)
        
        return (part_small + part_half + part_large) % mod