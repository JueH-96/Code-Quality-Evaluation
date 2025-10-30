class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        results = []

        def get_powerful_array(n):
            powerful_array = []
            power = 1
            temp_n = n
            while temp_n > 0:
                if temp_n & 1:
                    powerful_array.append(power)
                power *= 2
                temp_n >>= 1
            return powerful_array

        def length_powerful_array(n):
            return bin(n).count('1')

        def get_big_nums_element(index):
            count = 0
            n = 1
            while True:
                len_pa = length_powerful_array(n)
                if count + len_pa > index:
                    pa = get_powerful_array(n)
                    return pa[index - count]
                count += len_pa
                n += 1

        for from_i, to_i, mod_i in queries:
            product = 1
            for j in range(from_i, to_i + 1):
                element = get_big_nums_element(j)
                product = (product * element) % mod_i
            results.append(product)
        return results