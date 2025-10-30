class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_bits(n):
            # Count total number of set bits from 1 to n
            if n <= 0:
                return 0
            
            total = 0
            power = 1
            while power <= n:
                # Count complete groups of 2^(power+1)
                complete_groups = n // (power * 2)
                total += complete_groups * power
                
                # Count remaining bits
                remainder = n % (power * 2)
                if remainder >= power:
                    total += remainder - power + 1
                
                power *= 2
            return total
        
        def find_nth_element(n):
            # Find the nth element in big_nums (0-indexed)
            if n < 0:
                return 0
            
            # Binary search to find which number this index belongs to
            left, right = 1, n + 2
            while left < right:
                mid = (left + right) // 2
                if count_bits(mid) <= n:
                    left = mid + 1
                else:
                    right = mid
            
            num = left
            # Find position within this number's powerful array
            prev_count = count_bits(num - 1)
            pos_in_num = n - prev_count
            
            # Find which bit this corresponds to
            bit_pos = 0
            for i in range(60):
                if num & (1 << i):
                    if bit_pos == pos_in_num:
                        return 1 << i
                    bit_pos += 1
            
            return 1
        
        def product_range(from_idx, to_idx, mod):
            # Calculate product of big_nums[from_idx] to big_nums[to_idx] mod mod
            result = 1
            
            # Find which numbers the range spans
            left, right = 1, to_idx + 2
            while left < right:
                mid = (left + right) // 2
                if count_bits(mid) <= from_idx:
                    left = mid + 1
                else:
                    right = mid
            start_num = left
            
            left, right = 1, to_idx + 2
            while left < right:
                mid = (left + right) // 2
                if count_bits(mid) <= to_idx:
                    left = mid + 1
                else:
                    right = mid
            end_num = left
            
            # If range is within same number or adjacent numbers, calculate directly
            if end_num - start_num <= 1:
                for i in range(from_idx, to_idx + 1):
                    result = (result * find_nth_element(i)) % mod
                return result
            
            # Calculate contribution from each number
            # First number (partial)
            start_pos = from_idx - count_bits(start_num - 1)
            bits = bin(start_num)[2:]
            for i in range(start_pos, len(bits)):
                if bits[len(bits) - 1 - i] == '1':
                    result = (result * pow(2, i, mod)) % mod
            
            # Middle numbers (complete)
            for num in range(start_num + 1, end_num):
                bits = bin(num)[2:]
                for i in range(len(bits)):
                    if bits[len(bits) - 1 - i] == '1':
                        result = (result * pow(2, i, mod)) % mod
            
            # Last number (partial)
            end_pos = to_idx - count_bits(end_num - 1)
            bits = bin(end_num)[2:]
            for i in range(end_pos + 1):
                if bits[len(bits) - 1 - i] == '1':
                    result = (result * pow(2, i, mod)) % mod
            
            return result
        
        answer = []
        for from_i, to_i, mod_i in queries:
            result = product_range(from_i, to_i, mod_i)
            answer.append(result)
        
        return answer