import math

class Solution:
  def largestPalindrome(self, n: int, k: int) -> str:
    # Length of the first half of the palindrome.
    # E.g., n=5, L=3. n=4, L=2.
    L = (n + 1) // 2 

    # Precompute powers of 10 modulo k: p10[i] = 10^i % k
    # Max index needed for p10 is n-1.
    p10 = [0] * n 
    # Constraint: 1 <= n <= 10^5. So n cannot be 0.
    
    p10[0] = 1 % k
    for i in range(1, n):
      p10[i] = (p10[i-1] * 10) % k

    # Precompute coefficients C[j] for P_j (the j-th digit, 0-indexed, of the first half)
    # Palindrome D_0 D_1 ... D_{n-1}. P_j refers to D_j for j < L.
    # Value = sum( D_i * 10^(n-1-i) ) for i = 0 to n-1.
    # Modulo k, this is sum_{j=0}^{L-1} ( P_j * C[j] ).
    C = [0] * L
    for j in range(L):
      power_idx_for_left_digit = n - 1 - j # D_j is P_j, its term involves 10^(n-1-j)
      power_idx_for_right_digit = j        # D_{n-1-j} is also P_j, its term involves 10^j

      if power_idx_for_left_digit == power_idx_for_right_digit: 
        # This is the middle digit: n is odd and j = (n-1)/2.
        # It's D_j, which is P_j. It only appears once.
        C[j] = p10[power_idx_for_left_digit]
      else:
        # P_j corresponds to two digits D_j and D_{n-1-j}.
        C[j] = (p10[power_idx_for_left_digit] + p10[power_idx_for_right_digit]) % k

    # DP table: dp[idx][rem] is True if it's possible to choose digits P_idx...P_{L-1}
    # (suffix of the first half starting at P_idx)
    # such that their combined weighted sum modulo k is 'rem'.
    dp = [[False for _ in range(k)] for _ in range(L + 1)]

    # Base case: For an empty suffix of digits (idx == L), 
    # the sum is 0. So dp[L][0] is true.
    dp[L][0] = True

    # Fill DP table bottom-up for idx from L-1 down to 0
    for idx in range(L - 1, -1, -1):
      min_digit_val = 0
      if idx == 0: # P_0 must be >= 1 (no leading zeros; positive integer)
        min_digit_val = 1
      
      for current_rem_to_achieve in range(k):
        for d in range(min_digit_val, 10): # Digit P_idx takes value d
          # Remainder that P_{idx+1}...P_{L-1} (the rest of the suffix) must sum to
          rem_needed_from_rest_of_suffix = (current_rem_to_achieve - d * C[idx]) % k
          # Python's % handles negative results correctly for positive k (e.g. (-5)%3 = 1)
          
          if dp[idx+1][rem_needed_from_rest_of_suffix]:
            dp[idx][current_rem_to_achieve] = True
            break # Found a digit d; P_idx...P_{L-1} can sum to current_rem_to_achieve
    
    # If dp[0][0] is False, no N-digit palindrome is divisible by K.
    # This should not happen for k in [1,9] as k...k (N times) is a valid N-digit palindrome divisible by k.
    if not dp[0][0]:
        return "" 

    # Reconstruct the first half (ans_digits) to form the largest number
    ans_digits = []
    # The total sum of weighted digits P_0...P_{L-1} must be 0 mod k.
    current_rem_target_for_suffix = 0 

    for idx in range(L):
      min_digit_val = 0
      if idx == 0:
        min_digit_val = 1
      
      # Iterate digit d from 9 down to find the largest valid P_idx
      for d in range(9, min_digit_val - 1, -1):
        rem_needed_from_further_suffix = (current_rem_target_for_suffix - d * C[idx]) % k
        
        if dp[idx+1][rem_needed_from_further_suffix]:
          ans_digits.append(d)
          current_rem_target_for_suffix = rem_needed_from_further_suffix
          break # Found largest d for P_idx, move to P_{idx+1}
          
    # Construct the full palindrome string
    first_half_str = "".join(map(str, ans_digits))
    
    second_half_to_append_str = ""
    if n % 2 == 0: # n is even, second half is full reverse of first half
      second_half_to_append_str = first_half_str[::-1]
    else: # n is odd, second half is reverse of first half excluding its last char (the middle digit)
      second_half_to_append_str = first_half_str[:-1][::-1]
      
    return first_half_str + second_half_to_append_str