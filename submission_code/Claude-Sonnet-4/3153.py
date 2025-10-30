class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Count frequency of each bit position
        bit_count = [0] * 32  # 32 bits should be enough for 10^9
        
        for num in nums:
            for i in range(32):
                if num & (1 << i):
                    bit_count[i] += 1
        
        # Create k numbers by greedily assigning bits
        result_nums = [0] * k
        
        # For each bit position (from most significant to least)
        for bit_pos in range(31, -1, -1):
            count = bit_count[bit_pos]
            # Assign this bit to the smallest 'count' numbers among our k numbers
            # But we can assign at most k bits of this position
            assignments = min(count, k)
            
            # Sort to always assign to the currently smallest numbers
            result_nums.sort()
            
            for i in range(assignments):
                result_nums[i] |= (1 << bit_pos)
        
        # Calculate sum of squares
        total = 0
        for num in result_nums:
            total = (total + num * num) % MOD
        
        return total