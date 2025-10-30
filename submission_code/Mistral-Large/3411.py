from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def powerful_array(x):
            result = []
            power = 1
            while x > 0:
                if x % 2 == 1:
                    result.append(power)
                x //= 2
                power *= 2
            return result

        def generate_big_nums(limit):
            big_nums = []
            index = 0
            i = 1
            while index < limit:
                powerful = powerful_array(i)
                big_nums.extend(powerful)
                index += len(powerful)
                i += 1
            return big_nums

        max_index = max(q[1] for q in queries)
        big_nums = generate_big_nums(max_index + 1)

        results = []
        for from_i, to_i, mod_i in queries:
            product = 1
            for j in range(from_i, to_i + 1):
                product = (product * big_nums[j]) % mod_i
            results.append(product)

        return results