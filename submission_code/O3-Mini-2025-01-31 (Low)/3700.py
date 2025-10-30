from typing import List
from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        
        # Helper functions for combinations for small k.
        def comb(n_, k):
            if k < 0 or k > n_:
                return 0
            if k == 0:
                return 1
            if k == 1:
                return n_ % MOD
            if k == 2:
                return (n_ * (n_ - 1) // 2) % MOD
        
        # Given a frequency dictionary freq (for indices on one side) for non-a elements,
        # compute number of ways to choose k items
        # If distinct_required is True, then if k==2 we must choose from two different groups.
        # For k==1 or 0 the answer is the same as "choose any".
        def ways_non(freq: dict, k: int, distinct_required: bool) -> int:
            # total count of indices on that side:
            total = sum(freq.values())
            if k == 0:
                return 1
            if k == 1:
                return total % MOD
            if k == 2:
                if not distinct_required:
                    return comb(total, 2)
                else:
                    # choose two indices that come from different groups.
                    keys = list(freq.keys())
                    keys.sort()
                    s = 0
                    m = len(keys)
                    # sum_{i<j} freq[i] * freq[j]
                    prefix = 0
                    for v in keys:
                        s = (s + prefix * freq[v]) % MOD
                        prefix = (prefix + freq[v]) % MOD
                    return s % MOD
            return 0  # should not happen

        # We'll build prefix frequency maps as we iterate candidate j.
        # prefix_freq[j] will be frequency dictionary for indices [0, j) (all indices less than j)
        prefix_freq = []
        curr = defaultdict(int)
        for i in range(n):
            prefix_freq.append(dict(curr))
            curr[nums[i]] += 1
        # For suffix, we build similarly: suffix_freq[j] holds frequencies for indices (j, end)
        suffix_freq = [None] * n
        curr = defaultdict(int)
        for i in range(n-1, -1, -1):
            curr[nums[i]] += 1
            suffix_freq[i] = dict(curr)
        
        # Now iterate candidate j
        for j in range(n):
            a = nums[j]
            # Left side indices: 0 .. j-1, right side indices: j+1 .. n-1.
            L = j
            R = n - j - 1
            # frequency dictionaries for left non–a and right non–a:
            left_freq = {}
            for key, cnt in prefix_freq[j].items():
                if key != a:
                    left_freq[key] = cnt
            right_full = suffix_freq[j]
            right_freq = {}
            # right_full includes nums[j] itself but j is current so subtract one occurrence of a:
            # Actually suffix_freq[j] counts indices j..end so we need to remove the candidate at j.
            # But since candidate j is a and we only want indices > j, we do:
            for key, cnt in right_full.items():
                # We want indices strictly greater than j
                if key == a:
                    # subtract the one at index j (if exists) – but note: in suffix_freq[j] the current index j is counted.
                    if cnt - 1 > 0:
                        right_freq[key] = cnt - 1
                    # else, skip
                else:
                    right_freq[key] = cnt
            
            # Count of a in left and right
            L_a = prefix_freq[j].get(a, 0)
            R_a = right_full.get(a, 0) - 1  # since candidate j is a, remove one
            
            # total non–a count on each side:
            total_left_non = L - L_a
            total_right_non = (R if R>0 else 0) - (right_full.get(a,0)-1 if a in right_full else 0)
            
            # Now we consider four cases by f = frequency of a in subsequence = x + 1
            # where x (the extra copies chosen from left+right) goes 1,2,3,4.
            # We split x as x_left (from left) and x_right (from right).
            res_for_j = 0
            
            # Case f = 2 (x = 1)   with distinct restriction on non–a selections.
            if 1 <= 1 <= (L_a + R_a):  # x=1 possible only if there is at least one extra a in left+right.
                # possible splits: (1,0) and (0,1)
                # Option A: (x_left, x_right) = (1, 0)
                waysA = 0
                if L_a >= 1:
                    left_part = comb(L_a, 1)  # choose one a on left
                    # left non–a: need to choose (2 - 1)=1 from left. (No distinct requirement for single pick.)
                    left_part = (left_part * ways_non(left_freq, 1, False)) % MOD
                    # Right: x_right=0, so choose 0 a; then need to choose (2 - 0)=2 from right.
                    right_part = comb(R_a, 0)  * ways_non(right_freq, 2, True) % MOD
                    waysA = (left_part * right_part) % MOD
                # Option B: (x_left, x_right) = (0, 1)
                waysB = 0
                if R_a >= 1:
                    left_part = comb(L_a, 0) * ways_non(left_freq, 2, True) % MOD
                    right_part = comb(R_a, 1) * ways_non(right_freq, 1, False) % MOD
                    waysB = (left_part * right_part) % MOD
                res_for_j = (res_for_j + waysA + waysB) % MOD
            
            # Case f = 3 (x = 2)   (non–a count = 5 - 3 = 2) no distinct restrictions.
            if 2 <= (L_a + R_a):
                ways_case = 0
                # Possibility (0,2)
                if R_a >= 2:
                    left_way = comb(L_a, 0) * ways_non(left_freq, 2, False) % MOD
                    right_way = comb(R_a, 2) * ways_non(right_freq, 0, False) % MOD  # choose 0 non–a
                    ways_case = (ways_case + left_way * right_way) % MOD
                # Possibility (1,1)
                if L_a >= 1 and R_a >= 1:
                    left_way = comb(L_a, 1) * ways_non(left_freq, 1, False) % MOD
                    right_way = comb(R_a, 1) * ways_non(right_freq, 1, False) % MOD
                    ways_case = (ways_case + left_way * right_way) % MOD
                # Possibility (2,0)
                if L_a >= 2:
                    left_way = comb(L_a, 2) * ways_non(left_freq, 0, False) % MOD  # choose 0 non–a =1 way
                    right_way = comb(R_a, 0) * ways_non(right_freq, 2, False) % MOD
                    ways_case = (ways_case + left_way * right_way) % MOD
                res_for_j = (res_for_j + ways_case) % MOD
            
            # Case f = 4 (x = 3)   (non–a count = 1) no distinct restrictions.
            if 3 <= (L_a + R_a):
                ways_case = 0
                # Possibility (1,2)  left x_left=1, right x_right=2
                if L_a >= 1 and R_a >= 2:
                    left_way = comb(L_a, 1) * ways_non(left_freq, (2 - 1), False) % MOD  # left non–a: 1 index
                    right_way = comb(R_a, 2) * ways_non(right_freq, (2 - 2), False) % MOD  # right non–a: 0
                    ways_case = (ways_case + left_way * right_way) % MOD
                # Possibility (2,1)
                if L_a >= 2 and R_a >= 1:
                    left_way = comb(L_a, 2) * ways_non(left_freq, (2 - 2), False) % MOD
                    right_way = comb(R_a, 1) * ways_non(right_freq, (2 - 1), False) % MOD
                    ways_case = (ways_case + left_way * right_way) % MOD
                res_for_j = (res_for_j + ways_case) % MOD

            # Case f = 5 (x = 4)   (non–a count = 0) no non–a picks needed.
            if 4 <= (L_a + R_a):
                if L_a >= 2 and R_a >= 2:
                    ways_case = comb(L_a, 2) * comb(R_a, 2) % MOD
                    res_for_j = (res_for_j + ways_case) % MOD
            
            ans = (ans + res_for_j) % MOD
        
        return ans