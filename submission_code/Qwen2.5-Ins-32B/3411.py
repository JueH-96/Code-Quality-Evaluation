from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def get_powerful_array_length(n):
            length = 0
            while n:
                length += 1
                n &= (n - 1)
            return length
        
        def get_powerful_array_sum(n):
            sum = 0
            while n:
                sum += n & -n
                n &= (n - 1)
            return sum
        
        def get_powerful_array_start(n):
            start = 0
            while n:
                start += get_powerful_array_length(n - 1)
                n &= (n - 1)
            return start
        
        def get_powerful_array_value(n, index):
            length = get_powerful_array_length(n)
            if index >= length:
                return 0
            return (n >> (length - index - 1)) & 1
        
        def get_product_modulo(query):
            from_i, to_i, mod_i = query
            from_i += 1
            to_i += 1
            product = 1
            while from_i < to_i:
                n = get_powerful_array_sum(to_i - 1)
                m = get_powerful_array_sum(from_i - 1)
                if n == m:
                    product = (product * (1 << (get_powerful_array_length(to_i - 1) - 1))) % mod_i
                    to_i -= 1
                else:
                    start = get_powerful_array_start(n)
                    end = get_powerful_array_start(m)
                    if from_i <= end:
                        product = (product * (1 << (get_powerful_array_length(from_i - 1) - 1))) % mod_i
                        from_i += 1
                    else:
                        index = from_i - start - 1
                        value = get_powerful_array_value(n, index)
                        product = (product * value) % mod_i
                        from_i += 1
            return product
        
        return [get_product_modulo(query) for query in queries]