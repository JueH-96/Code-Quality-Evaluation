from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # If the total sum is less than target, it's impossible.
        if sum(nums) < target:
            return -1
        
        # We'll maintain counts for powers 0 up to MAX_BIT-1.
        # Using MAX_BIT = 63 is safe (target < 2^31, and nums[i] <= 2^30).
        MAX_BIT = 63
        cnt = [0] * MAX_BIT
        for num in nums:
            # Since num is a power of two, its index is log2(num) = num.bit_length()-1.
            cnt[num.bit_length() - 1] += 1
        
        ans = 0
        # Process each bit from 0 to MAX_BIT-1.
        for i in range(MAX_BIT):
            # If the i-th bit in target is 1, then we need one coin of value 2^i.
            if (target >> i) & 1:
                cnt[i] -= 1
            
            # If the count is negative, we have a shortage at this bit.
            if cnt[i] < 0:
                # Try to borrow from a higher power coin.
                j = i + 1
                while j < MAX_BIT and cnt[j] == 0:
                    j += 1
                if j == MAX_BIT:
                    return -1  # Not possible to cover the shortage.
                # Split one coin from power j all the way down to i.
                while j > i:
                    cnt[j] -= 1         # Use one coin at power j.
                    cnt[j - 1] += 2      # Splitting gives two coins at one lower power.
                    ans += 1            # Each split counts as one operation.
                    j -= 1
            
            # Carry over any surplus coins (each pair of coins of 2^i can form one coin of 2^(i+1))
            if i < MAX_BIT - 1:
                cnt[i + 1] += cnt[i] // 2
        
        return ans