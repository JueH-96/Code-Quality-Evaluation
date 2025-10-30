from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count how many of each power of two we have
        cnt = [0] * 32
        for v in nums:
            b = v.bit_length() - 1
            cnt[b] += 1

        ops = 0
        carry = 0
        # Process bits from 2^0 up to 2^30
        for i in range(31):
            # add all 2^i pieces we currently have
            carry += cnt[i]
            # if target needs this bit
            if (target >> i) & 1:
                if carry > 0:
                    # use one 2^i
                    carry -= 1
                else:
                    # need to split a higher bit
                    j = i + 1
                    while j < 32 and cnt[j] == 0:
                        j += 1
                    if j == 32:
                        return -1
                    # split down from j to i
                    while j > i:
                        # take one 2^j, make two 2^(j-1)
                        cnt[j] -= 1
                        cnt[j-1] += 2
                        ops += 1
                        j -= 1
                    # now we have at least 2 of 2^i
                    carry += cnt[i]
                    carry -= 1  # use one for the target
            # move half of the leftover 2^i up to become 2^(i+1)
            carry //= 2

        return ops