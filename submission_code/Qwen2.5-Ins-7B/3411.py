class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def powerful_array(n):
            array = []
            while n > 0:
                for power in range(31, -1, -1):
                    if (n >> power) & 1:
                        array.append(2 ** power)
                        n -= 2 ** power
                        break
            return array
        
        def get_index(n):
            i = 1
            while True:
                if 2 ** (i + 1) - 1 >= n:
                    return i
                i += 1
        
        def get_powerful_array_index(n):
            return sum(2 ** i for i in range(get_index(n) + 1)) - 1 + powerful_array(n)[0]
        
        prefix_product = [1]
        for i in range(1, 1000000):
            prefix_product.append(prefix_product[-1] * (2 ** get_powerful_array_index(i)) % 1000000007)
        
        def query(from_i, to_i, mod):
            return prefix_product[to_i] * pow(prefix_product[from_i - 1], -1, mod) % mod
        
        return [query(from_i, to_i, mod) for from_i, to_i, mod in queries]