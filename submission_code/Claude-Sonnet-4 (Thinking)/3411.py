class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_set_bits_up_to_n(n):
            """Count total set bits in binary representations of numbers 1 to n"""
            if n <= 0:
                return 0
            
            count = 0
            bit_pos = 0
            
            while (1 << bit_pos) <= n:
                cycle_len = 1 << (bit_pos + 1)
                complete_cycles = (n + 1) // cycle_len
                remainder = (n + 1) % cycle_len
                
                count += complete_cycles * (1 << bit_pos)
                if remainder > (1 << bit_pos):
                    count += remainder - (1 << bit_pos)
                
                bit_pos += 1
            
            return count
        
        def get_element_at_index(idx):
            """Find the element at given index in big_nums array"""
            # Binary search to find which number's powerful array contains this index
            left, right = 1, idx + 2
            
            while left < right:
                mid = (left + right) // 2
                if count_set_bits_up_to_n(mid) > idx:
                    right = mid
                else:
                    left = mid + 1
            
            number = left
            bits_before = count_set_bits_up_to_n(number - 1)
            pos_in_array = idx - bits_before
            
            # Find the pos_in_array-th set bit in number (0-indexed)
            bit_idx = 0
            count = 0
            
            while True:
                if number & (1 << bit_idx):
                    if count == pos_in_array:
                        return 1 << bit_idx
                    count += 1
                bit_idx += 1
        
        def pow_mod(base, exp, mod):
            """Fast modular exponentiation"""
            result = 1
            base %= mod
            while exp > 0:
                if exp & 1:
                    result = (result * base) % mod
                exp >>= 1
                base = (base * base) % mod
            return result
        
        def count_power_contributions(from_i, to_i):
            """Count how many times each power of 2 appears in the range"""
            power_counts = {}
            
            for i in range(from_i, to_i + 1):
                element = get_element_at_index(i)
                power = 0
                temp = element
                while temp > 1:
                    temp >>= 1
                    power += 1
                
                power_counts[power] = power_counts.get(power, 0) + 1
            
            return power_counts
        
        def solve_query(from_i, to_i, mod_i):
            if to_i - from_i < 60:  # For small ranges, compute directly
                product = 1
                for i in range(from_i, to_i + 1):
                    element = get_element_at_index(i)
                    product = (product * element) % mod_i
                return product
            
            # For larger ranges, count power contributions
            power_counts = count_power_contributions(from_i, to_i)
            
            # Product is 2^(sum of power * count for each power)
            total_exp = 0
            for power, count in power_counts.items():
                total_exp += power * count
            
            return pow_mod(2, total_exp, mod_i)
        
        results = []
        for from_i, to_i, mod_i in queries:
            results.append(solve_query(from_i, to_i, mod_i))
        
        return results