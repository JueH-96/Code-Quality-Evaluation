from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        import sys
        n = len(nums)
        if n == 0:
            return 0
        
        # Find bits set in k and bits not set in k
        bits_in_k = []
        bits_not_in_k = []
        for bit in range(31):
            if k & (1 << bit):
                bits_in_k.append(bit)
            else:
                bits_not_in_k.append(bit)
        
        # Precompute next unset for each bit not in k
        next_unset = [[] for _ in range(32)]
        for bit in bits_not_in_k:
            current = -1
            for i in range(n-1, -1, -1):
                if (nums[i] & (1 << bit)) == 0:
                    current = i
                next_unset[bit].append(current)
            if current == -1:
                next_unset[bit].append(-1)
            else:
                next_unset[bit].append(next_unset[bit][-1])
            # Reverse the lists to make them forward-looking
            next_unset[bit].reverse()
        
        # Compute the minimum next unset position across all bits not in k
        min_next_unset = [sys.maxsize] * n
        for bit in bits_not_in_k:
            for i in range(n):
                if next_unset[bit][i] != -1:
                    min_next_unset[i] = min(min_next_unset[i], next_unset[bit][i])
        
        # Sliding window to find subarrays where AND >= k
        left = 0
        current_and = (1 << 31) - 1  # All bits set
        count = 0
        for right in range(n):
            current_and &= nums[right]
            while current_and < k:
                current_and ^= nums[left]
                left += 1
            # Check if min_next_unset[left] <= right
            if min_next_unset[left] <= right:
                count += right - left + 1
        return count