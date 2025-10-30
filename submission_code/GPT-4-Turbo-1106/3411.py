from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def get_powerful_array(x):
            powers = []
            power = 1
            while x > 0:
                if x % 2 == 1:
                    powers.append(power)
                x //= 2
                power *= 2
            return powers[::-1]
        
        def get_big_num_at(index):
            # Find the number that the index belongs to
            num = 1
            total_length = 0
            while True:
                powerful_array = get_powerful_array(num)
                if total_length + len(powerful_array) >= index:
                    return powerful_array[index - total_length - 1]
                total_length += len(powerful_array)
                num += 1
        
        def product_modulo_from_to(from_i, to_i, mod_i):
            product = 1
            for i in range(from_i, to_i + 1):
                product = (product * get_big_num_at(i + 1)) % mod_i
            return product
        
        answer = []
        for query in queries:
            from_i, to_i, mod_i = query
            answer.append(product_modulo_from_to(from_i, to_i, mod_i))
        return answer

# Example usage:
# sol = Solution()
# print(sol.findProductsOfElements([[1,3,7], [2,5,3], [7,7,4]])) # Output: [4, 2, 2]