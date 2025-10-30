from typing import List
import math

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        count = [0] * 31
        for num in nums:
            power = int(math.log2(num))
            count[power] += 1
        
        operations = 0
        
        for k in range(31):
            if (target >> k) & 1:
                if count[k] > 0:
                    count[k] -= 1
                else:
                    m = None
                    for m_candidate in range(k + 1, 31):
                        if count[m_candidate] > 0:
                            m = m_candidate
                            break
                    if m is None:
                        return -1
                    operations += m - k
                    count[m] -= 1
                    # Split m down to k, which adds 2^(m - k) to count[k], but we need to adjust the counts
                    # However, in our model, after splitting, we only need to account for the required k
                    # So we add 2 to k and subtract 1 (since we need one for the target)
                    # The intermediate steps are not tracked as they are not needed for higher bits
                    # This is a simplification to avoid tracking all intermediate splits
                    count[k] += 2  # After splitting m down to k, we get 2^ (m - k) counts at k
                    count[k] -= 1  # Use one of them for the current bit
        return operations