class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_set_bits(n, bit):
            # Count how many numbers from 1 to n have the 'bit'-th bit set
            if n < (1 << bit):
                return 0
            
            cycle = 1 << (bit + 1)
            complete = n // cycle
            remain = n % cycle
            
            count = complete * (1 << bit)
            if remain >= (1 << bit):
                count += remain - (1 << bit) + 1
            
            return count
        
        def sum_popcount(n):
            # Sum of popcounts (number of set bits) from 1 to n
            if n == 0:
                return 0
            
            total = 0
            for bit in range(60):
                if (1 << bit) > n:
                    break
                total += count_set_bits(n, bit)
            
            return total
        
        def find_number(idx):
            # Binary search to find which number contains index idx (0-based)
            left, right = 1, 1
            while sum_popcount(right) <= idx:
                right = min(right * 2, 10**18)
            
            while left < right:
                mid = (left + right) // 2
                if sum_popcount(mid) <= idx:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        def get_kth_set_bit(num, k):
            # Get the k-th set bit position in num (0-indexed)
            count = 0
            for bit in range(60):
                if num & (1 << bit):
                    if count == k:
                        return bit
                    count += 1
            return -1
        
        def count_bit_contributions(from_idx, to_idx):
            # Count how many times each bit appears in the range
            from_num = find_number(from_idx)
            to_num = find_number(to_idx)
            
            bit_count = [0] * 60
            
            if from_num == to_num:
                # All indices within same number
                prev_count = sum_popcount(from_num - 1) if from_num > 1 else 0
                start_bit_idx = from_idx - prev_count
                end_bit_idx = to_idx - prev_count
                
                for bit_idx in range(start_bit_idx, end_bit_idx + 1):
                    bit_pos = get_kth_set_bit(from_num, bit_idx)
                    bit_count[bit_pos] += 1
            else:
                # First number (partial)
                prev_count = sum_popcount(from_num - 1) if from_num > 1 else 0
                start_bit_idx = from_idx - prev_count
                num_bits = bin(from_num).count('1')
                
                for bit_idx in range(start_bit_idx, num_bits):
                    bit_pos = get_kth_set_bit(from_num, bit_idx)
                    bit_count[bit_pos] += 1
                
                # Middle numbers (complete)
                for bit in range(60):
                    bit_count[bit] += count_set_bits(to_num - 1, bit) - count_set_bits(from_num, bit)
                
                # Last number (partial)
                last_prev_count = sum_popcount(to_num - 1)
                end_bit_idx = to_idx - last_prev_count
                
                for bit_idx in range(end_bit_idx + 1):
                    bit_pos = get_kth_set_bit(to_num, bit_idx)
                    bit_count[bit_pos] += 1
            
            return bit_count
        
        result = []
        for from_i, to_i, mod_i in queries:
            bit_count = count_bit_contributions(from_i, to_i)
            
            product = 1
            for bit in range(60):
                if bit_count[bit] > 0:
                    base = (1 << bit) % mod_i
                    product = (product * pow(base, bit_count[bit], mod_i)) % mod_i
            
            result.append(product)
        
        return result