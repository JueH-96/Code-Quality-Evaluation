class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        max_c = 800  # Maximum possible set bits for numbers up to 2^800
        steps = [0] * (max_c + 2)
        steps[1] = 0  # Base case
        
        # Precompute steps for each c from 2 to max_c
        for c in range(2, max_c + 1):
            binary = bin(c)
            set_bits = binary.count('1')
            steps[c] = 1 + steps[set_bits]
        
        # Determine valid set bits counts (S)
        S = []
        if k >= 1:
            for c in range(1, max_c + 1):
                if steps[c] <= k - 1:
                    S.append(c)
        
        # Handle edge case where n is 1 (binary "1")
        if s == "1":
            return 0
        
        # Edge case for k=0
        if k == 0:
            return 0
        
        # Convert binary string to digits
        digits = [int(c) for c in s]
        L = len(digits)
        
        # Initialize DP tables for tight=0 and tight=1 states
        prev_tight0 = [0] * (L + 2)  # prev_tight0[cnt] is the count for tight=0 state
        prev_tight1 = [0] * (L + 2)  # prev_tight1[cnt] is the count for tight=1 state
        prev_tight1[0] = 1  # Initial state: no bits chosen, tight=1, cnt=0
        
        for i in range(L):
            curr_tight0 = [0] * (L + 2)
            curr_tight1 = [0] * (L + 2)
            current_bit = digits[i]
            
            # Process tight=0 state: can choose any bit (0 or 1)
            for cnt in range(L + 1):
                if prev_tight0[cnt]:
                    for bit in [0, 1]:
                        new_cnt = cnt + bit
                        if new_cnt > L + 1:
                            continue
                        curr_tight0[new_cnt] = (curr_tight0[new_cnt] + prev_tight0[cnt]) % MOD
            
            # Process tight=1 state: can only choose up to current_bit
            max_bit = current_bit
            for cnt in range(L + 1):
                if prev_tight1[cnt]:
                    for bit in range(max_bit + 1):
                        new_cnt = cnt + bit
                        new_tight = (bit == max_bit)
                        if new_tight:
                            curr_tight1[new_cnt] = (curr_tight1[new_cnt] + prev_tight1[cnt]) % MOD
                        else:
                            curr_tight0[new_cnt] = (curr_tight0[new_cnt] + prev_tight1[cnt]) % MOD
            
            # Update previous states for next iteration
            prev_tight0, prev_tight1 = curr_tight0, curr_tight1
        
        # Sum up all counts for valid set bits in S
        total = 0
        for c in S:
            if c <= L:
                total = (total + prev_tight0[c]) % MOD
        
        return total