class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        cnt = [0] * 31  # Exponents from 0 to 30
        total = 0
        
        for num in nums:
            total += num
            exponent = (num).bit_length() - 1
            cnt[exponent] += 1
        
        if total < target:
            return -1
        
        operations = 0
        carry = 0
        
        for i in range(31):
            cnt[i] += carry
            required = (target >> i) & 1
            
            available = cnt[i]
            if available < required:
                # Find the smallest j > i with cnt[j] > 0
                j = i + 1
                while j < 31 and cnt[j] == 0:
                    j += 1
                if j >= 31:
                    return -1
                # Split j down to i
                operations += (j - i)
                cnt[j] -= 1
                add = 1 << (j - i)
                cnt[i] += add
                # Update available after adding
                available = cnt[i]
            
            if available < required:
                return -1  # Shouldn't happen if previous steps correct
            
            remaining = available - required
            carry = remaining // 2
            cnt[i] = remaining % 2
        
        return operations