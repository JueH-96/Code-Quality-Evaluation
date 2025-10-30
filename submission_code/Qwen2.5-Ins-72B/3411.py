class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def powerful_array(x):
            result = []
            power = 1
            while x > 0:
                if x & 1:
                    result.append(power)
                x >>= 1
                power <<= 1
            return result
        
        def prefix_product_modulo(queries):
            max_query = max(max(q[1] for q in queries), 10**5)
            prefix_product = [1]
            current_index = 0
            current_product = 1
            current_mod = 10**9 + 7  # A large prime number to avoid overflow
            
            for i in range(1, max_query + 1):
                for power in powerful_array(i):
                    current_product = (current_product * power) % current_mod
                    current_index += 1
                    prefix_product.append(current_product)
            
            results = []
            for from_i, to_i, mod_i in queries:
                if from_i == 0:
                    result = prefix_product[to_i] % mod_i
                else:
                    result = (prefix_product[to_i] * pow(prefix_product[from_i - 1], -1, current_mod)) % mod_i
                results.append(result)
            
            return results
        
        return prefix_product_modulo(queries)