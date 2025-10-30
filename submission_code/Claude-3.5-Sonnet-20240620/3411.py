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

        def generate_big_nums(start, end):
            result = []
            for i in range(start, end + 1):
                result.extend(powerful_array(i))
            return result

        def product_mod(arr, mod):
            result = 1
            for num in arr:
                result = (result * num) % mod
            return result

        answer = []
        for from_i, to_i, mod_i in queries:
            big_nums_segment = generate_big_nums(from_i + 1, to_i + 1)
            answer.append(product_mod(big_nums_segment, mod_i))
        
        return answer