import math

class Solution:
  memo_phi = {}
  memo_count_set_bits_at_pos = {}
  memo_L_N = {}
  memo_sum_exp_all_N = {}
  memo_get_sum_exponents_prefix = {}

  def get_phi(self, n: int) -> int:
    state = n # Memo key for phi
    if state in self.memo_phi:
        return self.memo_phi[state]
    
    original_n = n
    result = n
    p = 2
    while p * p <= n:
      if n % p == 0:
        while n % p == 0:
          n //= p
        result -= result // p
      p += 1
    if n > 1:
      result -= result // n
    
    self.memo_phi[original_n] = result
    return result

  def count_set_bits_at_pos_k_up_to_N(self, N_inclusive: int, k: int) -> int:
    state = (N_inclusive, k)
    if state in self.memo_count_set_bits_at_pos:
        return self.memo_count_set_bits_at_pos[state]

    if N_inclusive < 0:
        self.memo_count_set_bits_at_pos[state] = 0
        return 0
    
    val_2_pow_k_plus_1 = 1 << (k + 1)
    val_2_pow_k = 1 << k
    
    num_full_cycles = N_inclusive // val_2_pow_k_plus_1
    count = num_full_cycles * val_2_pow_k
    
    remainder_len = N_inclusive % val_2_pow_k_plus_1
    count += max(0, remainder_len - val_2_pow_k + 1)

    self.memo_count_set_bits_at_pos[state] = count
    return count

  def count_total_set_bits_up_to_N(self, N_val: int) -> int:
    state = N_val # Memo key for L(N)
    if state in self.memo_L_N:
        return self.memo_L_N[state]
    if N_val <= 0:
        self.memo_L_N[state] = 0
        return 0

    ans = 0
    for k in range(50): # Max bit position around log2(7e13) ~ 46. 50 is safe.
      ans += self.count_set_bits_at_pos_k_up_to_N(N_val, k)
    
    self.memo_L_N[state] = ans
    return ans

  def sum_exponents_for_all_numbers_up_to_N(self, N_target: int, P_val: int, C_val: int):
    state = (N_target, P_val, C_val)
    if state in self.memo_sum_exp_all_N:
        return self.memo_sum_exp_all_N[state]

    if N_target <= 0:
      self.memo_sum_exp_all_N[state] = (0,0)
      return (0, 0)
    
    s_mod_P = 0
    s_cap_C = 0
    
    for k in range(50): 
      count_k_set = self.count_set_bits_at_pos_k_up_to_N(N_target, k)
      if count_k_set == 0 and k > 0 and (1 << (k-1)) > N_target : # Opt: if k is larger than N_target's MSB
          break 

      term_mod_P = (k * (count_k_set % P_val)) % P_val
      s_mod_P = (s_mod_P + term_mod_P) % P_val
      
      if s_cap_C < C_val:
          term_for_cap = k * count_k_set # Python handles large integers
          s_cap_C += term_for_cap
          if s_cap_C >= C_val:
            s_cap_C = C_val
        
    self.memo_sum_exp_all_N[state] = (s_mod_P, s_cap_C)
    return (s_mod_P, s_cap_C)

  def get_sum_exponents_prefix(self, K_idx: int, P_val: int, C_val: int):
    state = (K_idx, P_val, C_val)
    if state in self.memo_get_sum_exponents_prefix:
        return self.memo_get_sum_exponents_prefix[state]

    if K_idx < 0:
      self.memo_get_sum_exponents_prefix[state] = (0,0)
      return (0,0)
    
    N0 = -1; low = 1; high = 10**14 # Max N0 ~7e13 for K_idx=10^15. 1e14 is safe bound.
    
    ans_N0 = 1 # Default N0 to 1 if K_idx is small (e.g. 0)
    while low <= high:
      mid = low + (high - low) // 2
      if mid == 0: low = 1; continue # Should not happen with low=1
      len_at_mid = self.count_total_set_bits_up_to_N(mid)
      if len_at_mid > K_idx:
        ans_N0 = mid
        high = mid - 1
      else:
        low = mid + 1
    N0 = ans_N0
    
    res_sum_exp_N0_minus_1 = self.sum_exponents_for_all_numbers_up_to_N(N0 - 1, P_val, C_val)
    current_sum_mod_P = res_sum_exp_N0_minus_1[0]
    current_sum_cap_C = res_sum_exp_N0_minus_1[1]
    
    len_up_to_N0_minus_1 = self.count_total_set_bits_up_to_N(N0 - 1)
    num_remaining_elements = K_idx - len_up_to_N0_minus_1 + 1
    
    for p in range(50): 
      if num_remaining_elements == 0: break
      if (N0 >> p) & 1: 
        current_sum_mod_P = (current_sum_mod_P + p) % P_val
        if current_sum_cap_C < C_val:
            current_sum_cap_C += p
            if current_sum_cap_C >= C_val:
              current_sum_cap_C = C_val
        num_remaining_elements -= 1
        
    self.memo_get_sum_exponents_prefix[state] = (current_sum_mod_P, current_sum_cap_C)
    return (current_sum_mod_P, current_sum_cap_C)

  def findProductsOfElements(self, queries: list[list[int]]) -> list[int]:
    ans_list = []
    
    self.memo_phi.clear()
    self.memo_count_set_bits_at_pos.clear()
    self.memo_L_N.clear()
    self.memo_sum_exp_all_N.clear()
    self.memo_get_sum_exponents_prefix.clear()

    for F_orig, T_orig, M_orig in queries:
      if M_orig == 1:
        ans_list.append(0)
        continue
      
      F_0idx, T_0idx = F_orig, T_orig # Query indices are 0-based per problem example
      
      P_M = self.get_phi(M_orig)
      # Cap C_M needs to be large enough to distinguish E < P_M from E >= P_M
      # Max sum of exponents for one N is ~1128 (for N with ~48 bits all set).
      # Max individual exponent p is ~48.
      # C_M = P_M + safety_margin. Safety margin should be at least max_sum_exponents_for_one_N0.
      C_M = P_M + 1200 
      
      sT, cT = self.get_sum_exponents_prefix(T_0idx, P_M, C_M)
      sF, cF = self.get_sum_exponents_prefix(F_0idx - 1, P_M, C_M)
      
      exp_val_mod_P_M = (sT - sF + P_M) % P_M
      
      # E_approx = S(T_0idx)_capped_at_C_M - S(F_0idx-1)_capped_at_C_M
      E_approx_using_caps = cT - cF
      
      final_exponent = exp_val_mod_P_M
      if E_approx_using_caps >= P_M :
        final_exponent += P_M
        
      query_ans = pow(2, final_exponent, M_orig)
      ans_list.append(query_ans)
      
    return ans_list