class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def powerful_array(x):
            powers = []
            while x > 0:
                highest_power = 1 << (x.bit_length() - 1)
                powers.append(highest_power)
                x -= highest_power
            return powers

        def big_nums_index(index):
            if index == 0:
                return 1
            total_length = 0
            n = 1
            while total_length <= index:
                total_length += len(powerful_array(n))
                n += 1
            n -= 1
            total_length -= len(powerful_array(n))
            remaining = index - total_length
            return powerful_array(n)[remaining]

        def product_modulo(from_i, to_i, mod_i):
            result = 1
            for i in range(from_i, to_i + 1):
                result = (result * big_nums_index(i)) % mod_i
            return result

        answer = []
        for query in queries:
            from_i, to_i, mod_i = query
            answer.append(product_modulo(from_i, to_i, mod_i))

        return answer