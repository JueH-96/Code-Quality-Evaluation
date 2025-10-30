from math import log2, floor

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def powerful_array(x):
            arr = []
            while x > 0:
                power = int(floor(log2(x)))
                arr.append(2**power)
                x -= 2**power
            return arr[::-1]

        def product_of_range(start, end):
            product = 1
            for i in range(start, end + 1):
                product *= powerful_array(i)[-1]
            return product

        answer = []
        for query in queries:
            start, end, mod = query
            product = product_of_range(start, end)
            answer.append(product % mod)
        return answer