class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def countSetBits(n, bit_pos):
            # Count how many numbers from 1 to n have bit at position bit_pos set
            if n < 0:
                return 0
            
            cycle_len = 1 << (bit_pos + 1)  # 2^(bit_pos + 1)
            complete_cycles = (n + 1) // cycle_len
            remainder = (n + 1) % cycle_len
            
            # In each complete cycle, half the numbers have this bit set
            count = complete_cycles * (cycle_len // 2)
            
            # For the incomplete cycle, check if we're in the second half
            if remainder > cycle_len // 2:
                count += remainder - cycle_len // 2
            
            return count
        
        def sumOfPrices(n):
            # Calculate sum of prices for all numbers from 1 to n
            total = 0
            bit_pos = 0
            
            while (1 << bit_pos) <= n:
                # Check if this bit position contributes to price
                # bit_pos is 0-indexed, but we need 1-indexed for the condition
                if (bit_pos + 1) % x == 0:
                    total += countSetBits(n, bit_pos)
                bit_pos += 1
            
            return total
        
        # Binary search for the answer
        left, right = 1, 10**15
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if sumOfPrices(mid) <= k:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result