from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # If total sum is less than target, it's impossible
        total = sum(nums)
        if total < target:
            return -1

        # Count how many powers of two we have at each bit position
        MAXB = 31  # enough for bits 0..30
        cnt = [0] * MAXB
        for x in nums:
            b = x.bit_length() - 1
            cnt[b] += 1

        ans = 0
        # 'available' is how many 2^i-"units" we can use at this bit level,
        # including both cnt[i] and pairs carried up from lower bits.
        available = 0

        for i in range(MAXB):
            # Add in all coins of value 2^i
            available += cnt[i]

            # If target has a 1 at bit i, we need one 2^i unit
            if (target >> i) & 1:
                if available > 0:
                    # Use one available unit
                    available -= 1
                else:
                    # Need to split a larger coin
                    j = i + 1
                    # Find next bit j>i where we have a coin
                    while j < MAXB and cnt[j] == 0:
                        j += 1
                    if j == MAXB:
                        # No larger coin to split -> impossible
                        return -1
                    # Split down from j to i
                    while j > i:
                        cnt[j] -= 1
                        cnt[j - 1] += 2
                        ans += 1
                        j -= 1
                    # Now we've produced two 2^i coins
                    available += cnt[i]
                    # Use one
                    available -= 1

            # Any leftover units at bit i can form pairs to become bit (i+1)
            available //= 2

        return ans