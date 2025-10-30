from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_set_bits_k(n, k):
            block_size = 2**(k+1)
            full_blocks_count = n // block_size
            set_bits_full_blocks = full_blocks_count * (2**k)
            remainder = n % block_size
            set_bits_remainder = max(0, remainder - (2**k) + 1)
            return set_bits_full_blocks + set_bits_remainder

        def cum_len_efficient(n):
            if n <= 0:
                return 0
            total_set_bits = 0
            num_bits = n.bit_length()
            for k in range(num_bits):
                total_set_bits += count_set_bits_k(n, k)
            return total_set_bits

        def powerful_array(x):
            powers = []
            power = 1
            while x > 0:
                if x % 2 == 1:
                    powers.append(power)
                x //= 2
                power *= 2
            return powers

        def get_big_nums_element(idx):
            low = 1
            high = 2 * 10**16  # Increased upper bound a bit
            number = -1
            index_in_powerful_array = -1
            while low <= high:
                mid = (low + high) // 2
                current_cum_len = cum_len_efficient(mid)
                prev_cum_len = cum_len_efficient(mid - 1) if mid > 1 else 0
                if current_cum_len > idx and (mid == 1 or prev_cum_len <= idx):
                    number = mid
                    index_in_powerful_array = idx - prev_cum_len
                    break
                elif current_cum_len <= idx:
                    low = mid + 1
                else:
                    high = mid - 1
            if number != -1:
                powerful_arr = powerful_array(number)
                return powerful_arr[index_in_powerful_array]
            return 0 # Should not reach here, but for safety

        answers = []
        for query in queries:
            from_i, to_i, mod_i = query
            product = 1
            for idx in range(from_i, to_i + 1):
                element = get_big_nums_element(idx - 1) # Adjust index to 0-based
                product = (product * element) % mod_i
            answers.append(product)
        return answers