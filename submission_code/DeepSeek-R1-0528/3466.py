class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        next_occurrence = [[n] * n for _ in range(32)]
        
        for b in range(32):
            next_occ = n
            for i in range(n-1, -1, -1):
                if (nums[i] >> b) & 1 == 0:
                    next_occ = i
                next_occurrence[b][i] = next_occ
        
        result = 0
        for i in range(n):
            j = i
            current_AND = nums[i]
            while j < n:
                next_change = n
                temp = current_AND
                while temp:
                    low_bit = temp & -temp
                    bit_index = low_bit.bit_length() - 1
                    next_change = min(next_change, next_occurrence[bit_index][j])
                    temp &= temp - 1
                if current_AND == k:
                    result += (next_change - j)
                j = next_change
                if j < n:
                    current_AND &= nums[j]
        return result