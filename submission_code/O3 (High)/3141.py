from typing import List
from collections import defaultdict

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total = sum(nums)                       # sum of one whole copy
        rem   = target % total                  # wanted remainder
        k0    = target // total                 # minimum number of whole copies that have to be taken

        # ----- a quick answer when only whole copies are needed -----
        answer = float('inf')
        if rem == 0:                            # all we need are whole copies of nums
            answer = k0 * n                     # (will be compared with other possibilities later)

        # -----------------------------------------------------------------
        # Any optimal sub-array is       suffix_of_some_copy
        #                           +   (some whole copies) *
        #                           +   prefix_of_following_copy
        #
        # The two coloured parts together are completely contained inside
        # two consecutive copies of `nums`, i.e. inside  `nums + nums`.
        #
        # Therefore it is enough to search inside the array  arr = nums + nums
        # for all sub-arrays whose sum  s  satisfies
        #            s <= target  and  (target - s) is a multiple of total
        #
        # While scanning the (at most 2·n) prefix sums we keep, for every
        # possible remainder r (0 … total-1) and for every possible
        # quotient q = ⌊prefixSum/total⌋ (which is at most 2) the *largest*
        # index where this (r,q) pair occurred.
        #
        # For the current prefix (i) we need a previous prefix (j) with
        #          (prefix[i] - prefix[j]) ≡ rem (mod total)
        # Let
        #          diff  = prefix[i] - prefix[j]
        # Then  target - diff  whole copies of  nums  still have to be added
        # and the complete length becomes
        #          (i - j) + ((target - diff)//total) · n
        # -----------------------------------------------------------------

        arr = nums + nums
        prefix_sum = 0
        prefix_list = [0]                       # prefix_list[index]  stores sum of first 'index' elements

        # best[residue] -> list with three entries for q = 0,1,2
        best = defaultdict(lambda: [-1, -1, -1])
        best[0][0] = 0                          # prefix index 0 has (remainder 0, quotient 0)

        for idx, val in enumerate(arr):         # idx   ranges   0 … 2n-1
            prefix_sum += val
            prefix_list.append(prefix_sum)
            i = idx + 1                         # current prefix index   (because prefix_list has an extra 0 at front)

            q_i = prefix_sum // total           # quotient and remainder of current prefix
            r_i = prefix_sum - q_i * total      #  (faster than  prefix_sum % total )

            # residue that a previous prefix must have
            r_need = (r_i - rem) % total

            # try every possible q of that previous prefix (0,1,2)
            prev_q_candidates = best.get(r_need, [-1, -1, -1])
            for q_prev in range(3):
                j = prev_q_candidates[q_prev]
                if j == -1:                      # we have never seen such a prefix yet
                    continue

                diff = prefix_sum - prefix_list[j]
                if diff > target:                # window sum too large
                    continue
                if (target - diff) % total != 0: # remaining part not multiple of total
                    continue

                copies_left = (target - diff) // total
                cur_len = (i - j) + copies_left * n
                if cur_len < answer:
                    answer = cur_len

            # after using the information of index i we store it for future steps
            if q_i <= 2:                         # q_i is never larger than 2 (we only scan two copies)
                if i > best[r_i][q_i]:           # we need the *largest* index for every (r,q)
                    best[r_i][q_i] = i

        return -1 if answer == float('inf') else answer