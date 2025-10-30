class Solution:
    memo: dict 
    S: str    
    k_divisor: int 
    L: int    
    max_c: int # Max count for ec or oc, L // 2. ec and oc cannot exceed this.

    def dp(self, idx: int, rem: int, ec: int, oc: int, is_less: bool, is_started: bool) -> int:
        # Pruning: if current counts of even/odd digits exceed L // 2.
        # If ec == oc, then total digits = 2*ec. This must be <= L. So ec <= L // 2.
        if ec > self.max_c or oc > self.max_c:
             return 0 
        
        # Base case: processed all digits
        if idx == self.L:
            # A number is beautiful if:
            # 1. It's a positive number (is_started is True)
            # 2. It's divisible by k_divisor (rem == 0)
            # 3. Count of even digits equals count of odd digits (ec == oc)
            # Note: If is_started is True and ec == oc == 0, it means no digits were counted,
            # which contradicts is_started. So, ec == oc implies ec > 0 if is_started.
            if is_started and rem == 0 and ec == oc:
                return 1
            else:
                return 0
        
        state = (idx, rem, ec, oc, is_less, is_started)
        if state in self.memo:
            return self.memo[state]

        res = 0
        # Determine the upper limit for the current digit
        limit = int(self.S[idx]) if not is_less else 9

        for digit in range(limit + 1):
            current_is_less = is_less or (digit < limit)
            
            if not is_started and digit == 0:
                # Leading zero: ec, oc, rem are not updated. is_started remains False.
                res += self.dp(idx + 1, 0, 0, 0, current_is_less, False)
            else:
                # Non-leading zero or non-zero digit: update ec, oc, rem. is_started becomes True.
                new_ec = ec + (1 if digit % 2 == 0 else 0)
                new_oc = oc + (1 if digit % 2 != 0 else 0)
                
                new_rem = (rem * 10 + digit) % self.k_divisor
                
                res += self.dp(idx + 1, new_rem, new_ec, new_oc, current_is_less, True)
        
        self.memo[state] = res
        return res

    def solve_for_num(self, num_str: str, k_val: int) -> int:
        # Helper to count beautiful numbers from 1 up to num_str (inclusive).
        
        # Handles num_str = "0" (from low - 1 when low = 1).
        # The DP itself correctly returns 0 for "0" because is_started will be False.
        if num_str == "0":
            return 0
            
        self.S = num_str
        self.k_divisor = k_val
        self.L = len(self.S)
        self.max_c = self.L // 2 
        self.memo = {} # Clear memoization table for this specific number
        
        return self.dp(0, 0, 0, 0, False, False)

    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        # Count beautiful integers up to high
        count_high = self.solve_for_num(str(high), k)
        
        # Count beautiful integers up to low - 1
        count_low_minus_1 = self.solve_for_num(str(low - 1), k)
        
        return count_high - count_low_minus_1