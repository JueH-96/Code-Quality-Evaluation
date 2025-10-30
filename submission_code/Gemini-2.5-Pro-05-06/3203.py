import collections

class Solution:
  def canMakePalindromeQueries(self, s: str, queries: list[list[int]]) -> list[bool]:
    n = len(s)
    half = n // 2

    # Precompute mismatch prefix sums
    # M[i] = 1 if s[i] != s[n-1-i], else 0, for i in [0, half-1)
    # mismatch_pref[k] stores sum of M[j] for j < k
    mismatch_pref = [0] * (half + 1)
    for i in range(half):
      mismatch_pref[i+1] = mismatch_pref[i]
      if s[i] != s[n-1-i]:
        mismatch_pref[i+1] += 1

    def query_mismatch_sum(L: int, R: int) -> int:
      if L > R: # handles L > R or R = -1 (e.g. L=0, R=-1)
        return 0
      # sum of M[i] for i in [L, R]
      # L, R are 0-indexed
      return mismatch_pref[R+1] - mismatch_pref[L]

    # Precompute character counts prefix sums for s1 = s[0...half-1]
    # s1_char_pref[k][char_code] = count of char_code in s[0...k-1]
    s1_char_pref = [[0] * 26 for _ in range(half + 1)]
    for i in range(half):
      for char_code in range(26):
        s1_char_pref[i+1][char_code] = s1_char_pref[i][char_code]
      s1_char_pref[i+1][ord(s[i]) - ord('a')] += 1

    # Precompute character counts prefix sums for s_rev, where s_rev[i] = s[n-1-i]
    # s_rev_char_pref[k][char_code] = count of char_code in s_rev[0...k-1]
    # s_rev[i] is character s[n-1-i]
    s_rev_chars_for_pref_sum = [s[n-1-i] for i in range(half)] 
    
    s_rev_char_pref = [[0] * 26 for _ in range(half + 1)]
    for i in range(half):
      for char_code in range(26):
        s_rev_char_pref[i+1][char_code] = s_rev_char_pref[i][char_code]
      s_rev_char_pref[i+1][ord(s_rev_chars_for_pref_sum[i]) - ord('a')] += 1

    def get_char_counts_vec(pref_arr: list[list[int]], L: int, R: int) -> list[int]:
      # Get char counts for range [L, R] (inclusive)
      if L > R:
        return [0] * 26
      
      res_counts = [0] * 26
      for char_code in range(26):
        res_counts[char_code] = pref_arr[R+1][char_code] - pref_arr[L][char_code]
      return res_counts

    ans = []
    for a_orig, b_orig, c_orig, d_orig in queries:
      # Query ranges: s[a_orig..b_orig] and s[c_orig..d_orig]
      # L0, R0 are for s[a_orig..b_orig] in first half
      L0, R0 = a_orig, b_orig
      # L1, R1 are indices in first half corresponding to s[c_orig..d_orig] in second half
      # s[c_orig..d_orig] means characters s[n-1-i] for i in [n-1-d_orig ... n-1-c_orig]
      # So we use s_rev[n-1-d_orig ... n-1-c_orig]
      L1, R1 = n - 1 - d_orig, n - 1 - c_orig

      # Part 1: Check outer mismatches
      # Mismatches before min(L0, L1)
      if query_mismatch_sum(0, min(L0, L1) - 1) > 0:
        ans.append(False)
        continue
      
      # Mismatches after max(R0, R1)
      if query_mismatch_sum(max(R0, R1) + 1, half - 1) > 0:
        ans.append(False)
        continue
        
      # Mismatches in the gap between [L0,R0] and [L1,R1] if they are disjoint
      if R0 < L1 - 1: # Gap is [R0+1, L1-1]
        if query_mismatch_sum(R0 + 1, L1 - 1) > 0:
          ans.append(False)
          continue
      if R1 < L0 - 1: # Gap is [R1+1, L0-1]
        if query_mismatch_sum(R1 + 1, L0 - 1) > 0:
          ans.append(False)
          continue
      
      # Part 2: Check inner character counts
      # Chars available in s[a_orig..b_orig]
      current_counts1 = get_char_counts_vec(s1_char_pref, L0, R0)
      # Chars available in s[c_orig..d_orig] (which is s_rev[L1..R1])
      current_counts2 = get_char_counts_vec(s_rev_char_pref, L1, R1)
      
      possible = True
      # Demands on current_counts1: characters s_rev[i] for i in ([L0,R0] \ [L1,R1])
      # [L0,R0] \ [L1,R1] = [L0, min(R0, L1-1)] U [max(L0, R1+1), R0]
      demands1A_chars = get_char_counts_vec(s_rev_char_pref, L0, min(R0, L1-1))
      demands1B_chars = get_char_counts_vec(s_rev_char_pref, max(L0, R1+1), R0)

      for char_code in range(26):
        current_counts1[char_code] -= (demands1A_chars[char_code] + demands1B_chars[char_code])
        if current_counts1[char_code] < 0:
          possible = False
          break
      if not possible:
        ans.append(False)
        continue
        
      # Demands on current_counts2: characters s[i] for i in ([L1,R1] \ [L0,R0])
      # [L1,R1] \ [L0,R0] = [L1, min(R1, L0-1)] U [max(L1, R0+1), R1]
      demands2A_chars = get_char_counts_vec(s1_char_pref, L1, min(R1, L0-1))
      demands2B_chars = get_char_counts_vec(s1_char_pref, max(L1, R0+1), R1)

      for char_code in range(26):
        current_counts2[char_code] -= (demands2A_chars[char_code] + demands2B_chars[char_code])
        if current_counts2[char_code] < 0:
          possible = False
          break
      if not possible:
        ans.append(False)
        continue

      # Part 3: Compare remaining counts
      # Remaining counts are for indices i in ([L0,R0] INTERSECT [L1,R1])
      if current_counts1 == current_counts2:
        ans.append(True)
      else:
        ans.append(False)
        
    return ans