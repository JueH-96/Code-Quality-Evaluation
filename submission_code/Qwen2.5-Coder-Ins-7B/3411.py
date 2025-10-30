class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        MOD = 10**9 + 7
        MAX_N = 10**15
        MAX_LOG = 60
        
        # Precompute the powerful arrays for numbers up to MAX_N
        powerful_arrays = [[] for _ in range(MAX_N + 1)]
        for i in range(1, MAX_N + 1):
            x = i
            while x > 0:
                if x & 1:
                    powerful_arrays[i].append(1 << (x.bit_length() - 1))
                x >>= 1
        
        # Flatten the list of powerful arrays into big_nums
        big_nums = []
        for array in powerful_arrays:
            big_nums.extend(array)
        
        # Precompute the prefix products modulo MOD
        prefix_products = [1]
        for num in big_nums:
            prefix_products.append((prefix_products[-1] * num) % MOD)
        
        # Answer each query
        result = []
        for from_i, to_i, mod_i in queries:
            result.append((prefix_products[to_i + 1] * pow(prefix_products[from_i], MOD - 2, MOD)) % mod_i)
        
        return result