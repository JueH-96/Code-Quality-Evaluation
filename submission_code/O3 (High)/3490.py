from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # counters of each parity  - they give the best length
        # when all chosen numbers have the same parity
        even_cnt = odd_cnt = 0

        # longest alternating-parity subsequence that
        # currently ends with an even / an odd value
        end_even = end_odd = 0

        for v in nums:
            if v & 1:                     # current number is odd
                odd_cnt += 1
                # either keep the old sequence that already ends with odd
                # or extend one that ended with an even number
                end_odd = max(end_odd, end_even + 1)
            else:                         # current number is even
                even_cnt += 1
                # either keep the old sequence that already ends with even
                # or extend one that ended with an odd number
                end_even = max(end_even, end_odd + 1)

        # 1) all-same-parity subsequence  ->  max(even_cnt, odd_cnt)
        # 2) alternating-parity subsequence -> max(end_even, end_odd)
        return max(max(even_cnt, odd_cnt), end_even, end_odd)