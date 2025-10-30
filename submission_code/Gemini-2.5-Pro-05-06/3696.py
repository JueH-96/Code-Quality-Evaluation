import collections

class Solution:
  def countSubstrings(self, s: str) -> int:
    n = len(s)
    ans = 0

    # For d=3,6,9 (sum-of-digits method)
    counts_mod3 = collections.defaultdict(int)
    counts_mod3[0] = 1
    current_pref_sum_mod3 = 0

    counts_mod9 = collections.defaultdict(int)
    counts_mod9[0] = 1
    current_pref_sum_mod9 = 0

    # For d=7 (suffix DP method)
    dp_suff_count_mod7 = collections.defaultdict(int)

    for j in range(n):
        char_sj = s[j]
        val_sj = int(char_sj)

        # --- Update running sums and DPs based on s[j] ---
        current_pref_sum_mod3 = (current_pref_sum_mod3 + val_sj) % 3
        current_pref_sum_mod9 = (current_pref_sum_mod9 + val_sj) % 9

        new_dp_suff_count_mod7 = collections.defaultdict(int)
        new_dp_suff_count_mod7[val_sj % 7] = 1
        
        # This loop correctly handles j=0 because dp_suff_count_mod7 is empty (all counts 0).
        for r_prev in range(7):
            if dp_suff_count_mod7[r_prev] > 0:
                r_curr = (r_prev * 10 + val_sj) % 7
                new_dp_suff_count_mod7[r_curr] += dp_suff_count_mod7[r_prev]
        dp_suff_count_mod7 = new_dp_suff_count_mod7
        
        # --- Add to ans based on s[j] as the last digit ---
        if char_sj == '0':
            counts_mod3[current_pref_sum_mod3] += 1
            counts_mod9[current_pref_sum_mod9] += 1
            continue

        d = val_sj

        if d == 1 or d == 2 or d == 5:
            ans += (j + 1)
        elif d == 3 or d == 6:
            ans += counts_mod3[current_pref_sum_mod3]
        elif d == 9:
            ans += counts_mod9[current_pref_sum_mod9]
        elif d == 4:
            ans += 1 
            if j > 0:
                if int(s[j-1]) % 2 == 0:
                    ans += j
        elif d == 8:
            ans += 1
            if j > 0:
                if int(s[j-1]) % 4 == 0:
                    ans += 1
            if j > 1:
                val_s_j_minus_2 = int(s[j-2])
                val_s_j_minus_1 = int(s[j-1])
                if (val_s_j_minus_2 * 4 + val_s_j_minus_1 * 2) % 8 == 0:
                    ans += (j - 1) 
        elif d == 7:
            ans += dp_suff_count_mod7[0]
        
        counts_mod3[current_pref_sum_mod3] += 1
        counts_mod9[current_pref_sum_mod9] += 1
            
    return ans