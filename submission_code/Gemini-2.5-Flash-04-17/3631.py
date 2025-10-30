import sys
# Increasing recursion depth is generally a good practice for deep recursion,
# although for L=800 and k=5, the state space is manageable and
# the depth of recursion in solve is L, which is 800.
# Standard limit is often 1000, so 800 is okay.
# sys.setrecursionlimit(2000)

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        L = len(s)
        MAX_POPCOUNT = L # Maximum possible popcount for a number < 2^L

        # Compute popcount(x) for x up to MAX_POPCOUNT
        pc = [0] * (MAX_POPCOUNT + 1)
        for i in range(1, MAX_POPCOUNT + 1):
            pc[i] = bin(i).count('1')

        # Compute popcount^i(v) for v in [1, MAX_POPCOUNT] and i in [0, k-1]
        # pc_power[v][i] = popcount^i(v)
        # We need this to determine if popcount^(k-1)(y) == 1 for potential popcount values y.
        pc_power = [[0] * k for _ in range(MAX_POPCOUNT + 1)]
        for v in range(1, MAX_POPCOUNT + 1):
            pc_power[v][0] = v
            for i in range(1, k):
                prev_val = pc_power[v][i-1]
                # As shown in thought process, for v <= 800 and i < k <= 5,
                # prev_val will be <= 800. The pc table is sufficient.
                if prev_val == 0: # Should not happen for positive v
                    pc_power[v][i] = 0
                else:
                    pc_power[v][i] = pc[prev_val]

        # The set target_vals = {y | 1 <= y <= MAX_POPCOUNT and y is (k-1)-reducible}
        # y is (k-1)-reducible iff popcount^(k-1)(y) == 1.
        target_vals = set()
        # Special case k=1: y is 0-reducible iff y == 1.
        # pc_power[y][1-1] = pc_power[y][0] = y. Condition is y == 1. target_vals = {1}.
        # This is correctly handled by the loop below.
        
        for y in range(1, MAX_POPCOUNT + 1):
             if pc_power[y][k-1] == 1:
                 target_vals.add(y)

        # Digit DP to count numbers x in [0, n] with popcount(x) in target_vals
        # The DP counts numbers represented by binary strings of length L which are <= s lexicographically.
        # This is equivalent to counting numbers x in [0, n].
        # We need [1, n-1]. Result = count([0, n]) - count(n) - count(0).
        # count(0) = 0 since 0 is not in target_vals (target_vals contains positive integers).

        memo = {}

        def solve(index, count, tight):
            state = (index, count, tight)
            if state in memo:
                return memo[state]

            if index == L:
                # Base case: finished building the number. 'count' is the total popcount.
                # Note: count will be <= L = MAX_POPCOUNT.
                return 1 if count in target_vals else 0

            ans = 0
            upper_bound = int(s[index]) if tight else 1

            for digit in range(upper_bound + 1):
                new_tight = tight and (digit == upper_bound)
                new_count = count + digit
                ans = (ans + solve(index + 1, new_count, new_tight)) % MOD

            memo[state] = ans
            return ans

        # Count numbers x in [0, n] with popcount(x) in target_vals
        count_up_to_n = solve(0, 0, True)

        # Check if n is k-reducible
        # n is k-reducible <=> popcount^k(n) == 1.
        
        current_val = s.count('1') # This is popcount^1(n)
        
        # Need to compute popcount^i(n) for i = 2 to k.
        # Start from popcount^1(n). Apply pc operation k-1 more times.
        # popcount^k(n) = pc(pc(...pc(n)...)) (k times)
        # Let v_0 = n, v_1 = pc(v_0), v_2 = pc(v_1), ..., v_k = pc(v_{k-1}).
        # We need to check if v_k == 1.
        # v_1 = s.count('1') <= L = MAX_POPCOUNT.
        # v_2 = pc(v_1) = pc[v_1] (since v_1 <= MAX_POPCOUNT)
        # v_i = pc[v_{i-1}] for i = 2 .. k.
        
        val_at_k_steps = s.count('1') # This is popcount^1(n)
        
        # Compute popcount^i(n) for i = 2 to k
        for _ in range(k - 1):
            if val_at_k_steps == 0: # Should not happen for n >= 1
                break
            
            # We need popcount of val_at_k_steps.
            # As analyzed, for n < 2^800, popcount(n) <= 800.
            # popcount^2(n) <= 9. popcount^3(n) <= 4. popcount^4(n) <= 1. popcount^5(n) <= 1.
            # For k <= 5, the intermediate values val_at_k_steps will always be <= 800.
            # The pc table is sufficient.
            
            val_at_k_steps = pc[val_at_k_steps] if val_at_k_steps <= MAX_POPCOUNT else bin(val_at_k_steps).count('1') # Fallback, shouldn't be needed

        is_n_reducible = (val_at_k_steps == 1)

        # Result = count([0, n]) - count(n)
        # count(0) is 0.
        ans = (count_up_to_n - (1 if is_n_reducible else 0) + MOD) % MOD

        return ans