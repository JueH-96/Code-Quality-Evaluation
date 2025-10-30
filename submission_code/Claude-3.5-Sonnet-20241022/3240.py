class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_price(n):
            if n <= 0:
                return 0
                
            # Convert to binary and pad with leading zeros
            bits = bin(n)[2:]
            length = len(bits)
            total = 0
            
            # For each position that is divisible by x
            for pos in range(1, length + 1):
                if pos % x != 0:
                    continue
                    
                # Get the digit at current position from right
                curr_pos = length - pos
                if curr_pos < 0:
                    break
                    
                # Count set bits at this position for all numbers up to n
                if curr_pos == 0:
                    total += (n + 1) // 2
                else:
                    # Get the number formed by bits before current position
                    prefix = int(bits[:curr_pos], 2) if curr_pos > 0 else 0
                    # Get the bit at current position
                    curr_bit = int(bits[curr_pos])
                    # Get remaining bits
                    suffix_len = length - curr_pos - 1
                    
                    # Calculate total set bits
                    total += prefix * (1 << suffix_len)
                    if curr_bit == 1:
                        total += (n % (1 << suffix_len)) + 1
            
            return total
            
        # Binary search for the answer
        left, right = 1, 10**18
        
        while left < right:
            mid = (left + right + 1) // 2
            if count_price(mid) <= k:
                left = mid
            else:
                right = mid - 1
                
        return left