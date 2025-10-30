import sys
# Setting a higher recursion depth might be necessary for very large inputs, 
# although N <= 10 is usually manageable with Python's default limit.
# sys.setrecursionlimit(2000) 

class Solution:
    
    def __init__(self):
        # Precompute L_prime once for the instance.
        # L_prime is the LCM of all integers S <= 81 such that the prime factors of S are in {2, 3, 5, 7}.
        # The maximum possible sum of digits for a number less than 10^9 (up to 9 digits) is 9*9 = 81.
        # L_prime = 2^6 * 3^4 * 5^2 * 7^2 = 64 * 81 * 25 * 49 = 6350400
        self.L_prime = 6350400
        # Memoization cache, will be reset for each call to solve
        self.memo = {}
        # Input string s and its length N, will be set in solve method
        self.s = ""
        self.N = 0

    def beautifulNumbers(self, l: int, r: int) -> int:
        """
        Counts the number of beautiful integers between l and r, inclusive.
        A positive integer is beautiful if the product of its digits is divisible by the sum of its digits.
        The approach uses digit DP (dynamic programming) with memoization.
        The count is computed as count(r) - count(l-1), where count(x) is the number
        of beautiful positive integers less than or equal to x.
        """
        
        s_r = str(r)
        # Calculate count for numbers <= r
        res_r = self.solve(s_r)
        
        # Calculate count for numbers <= l-1
        s_l_minus_1 = str(l - 1)
        res_l_minus_1 = self.solve(s_l_minus_1)
        
        # The result is the difference
        return res_r - res_l_minus_1

    def solve(self, s: str) -> int:
        """
        Helper function to count beautiful numbers less than or equal to the number represented by string s.
        Initializes memoization cache and sets up instance variables for the DP call.
        Handles the base case where s represents "0".
        """
        # If s is "0", there are no positive beautiful numbers <= 0.
        if s == "0":
             return 0
             
        self.memo = {} # Reset memoization cache for this new calculation
        self.s = s 
        self.N = len(s)
        
        # Initial call to the DP function.
        # State parameters: idx, tight, is_started, has_zero, current_sum, current_prod_mod
        # Initial product modulo L_prime is 1 (multiplicative identity).
        return self.dp_final(0, True, False, False, 0, 1)

    def check_large_prime_factor_inline(self, S):
        """ 
        Checks if the sum S has any prime factor greater than 7.
        This check is relevant only when the number has no zero digits.
        """
        # If S is 0, it means the number processed was 0 (or had sum 0 somehow),
        # which should be handled before this check. Safety check.
        if S == 0: return False 
        
        temp_S = S
        # Divide S by primes 2, 3, 5, 7 until it's no longer divisible by them.
        if temp_S % 2 == 0:
            while temp_S % 2 == 0: temp_S //= 2
        if temp_S % 3 == 0:
             while temp_S % 3 == 0: temp_S //= 3
        if temp_S % 5 == 0:
             while temp_S % 5 == 0: temp_S //= 5
        if temp_S % 7 == 0:
             while temp_S % 7 == 0: temp_S //= 7
        
        # If the remaining value of temp_S is greater than 1, it means S had a prime factor > 7.
        return temp_S > 1

    def dp_final(self, idx, tight, is_started, has_zero, current_sum, current_prod_mod):
        """
        Recursive DP function with memoization. This is the core logic.
        Parameters:
          idx: current digit index being processed (from left, 0-based)
          tight: boolean constraint; True if we are restricted by the digits of the upper bound string s
          is_started: boolean; True if a non-zero digit has been placed (to distinguish 0 from empty prefix)
          has_zero: boolean; True if a zero digit has been included in the number
          current_sum: the sum of digits placed so far
          current_prod_mod: the product of digits placed so far, modulo L_prime
        """
        
        # Base case: If we have processed all digits
        if idx == self.N:
            # If is_started is False, it means the number formed is 0. Not a positive integer.
            if not is_started: return 0 
            # If is_started is True, the number is positive and current_sum > 0.
            
            # If a zero digit was included, the product P is 0. 
            # Since sum S > 0, P % S == 0 % S == 0. Thus, the number is beautiful.
            if has_zero: return 1 
            
            # Case: No zero digit encountered. Product P is non-zero.
            # Check if current_sum is 0. Should not happen if is_started is True, but for safety.
            if current_sum == 0: return 0 
            
            # Check if the sum S has any prime factors greater than 7.
            # If yes, P cannot be divisible by S because P only has prime factors 2, 3, 5, 7. Not beautiful.
            if self.check_large_prime_factor_inline(current_sum):
                 return 0 
            
            # Case: No zero digit, and S only has prime factors from {2, 3, 5, 7}.
            # We need to check if P is divisible by S.
            # Since S divides L_prime in this case, P % S == 0 is equivalent to (P mod L_prime) % S == 0.
            if current_prod_mod % current_sum == 0:
                return 1 # Beautiful
            else:
                return 0 # Not beautiful

        # Memoization check: If this state has been computed before, return the stored result.
        state_key = (idx, tight, is_started, has_zero, current_sum, current_prod_mod)
        if state_key in self.memo:
            return self.memo[state_key]
        
        # Determine the upper limit for the current digit
        limit = int(self.s[idx]) if tight else 9
        ans = 0
        
        # Iterate through all possible digits (0 to limit) for the current position
        for digit in range(limit + 1):
            # Calculate the new tight constraint for the next recursive call
            new_tight = tight and (digit == limit)
            
            if not is_started:
                # If we haven't placed any non-zero digit yet
                if digit == 0:
                    # Still placing leading zeros. State remains largely unchanged.
                    # is_started=False, has_zero=False, sum=0, prod_mod=1
                    ans += self.dp_final(idx + 1, new_tight, False, False, 0, 1) 
                else:
                    # First non-zero digit encountered. Start tracking sum and product.
                    # is_started=True, has_zero=False, sum=digit, prod_mod=digit % L_prime
                    # Note: Since digit is 1..9, digit % L_prime is just digit.
                    ans += self.dp_final(idx + 1, new_tight, True, False, digit, digit % self.L_prime)
            else:
                # If we are already processing digits after the first non-zero one
                new_sum = current_sum + digit
                # Check if a zero digit is encountered now or has been encountered previously
                new_has_zero = has_zero or (digit == 0)
                
                if new_has_zero:
                    # If zero is encountered, the product becomes 0. 
                    # Set prod_mod to 0 permanently to reflect this.
                    new_prod_mod = 0 
                    ans += self.dp_final(idx + 1, new_tight, True, True, new_sum, new_prod_mod) 
                else: 
                    # Digit is non-zero, and no zero encountered so far. Update product modulo L_prime.
                    # current_prod_mod might be 0 if previous product was a multiple of L_prime. If so, it stays 0.
                    if current_prod_mod == 0:
                         new_prod_mod = 0
                    else:
                         new_prod_mod = (current_prod_mod * digit) % self.L_prime
                    ans += self.dp_final(idx + 1, new_tight, True, False, new_sum, new_prod_mod)
        
        # Store the computed result for this state in the memoization cache and return it.
        self.memo[state_key] = ans
        return ans