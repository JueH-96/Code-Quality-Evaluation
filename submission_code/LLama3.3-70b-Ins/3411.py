from typing import List

class Solution:
    def powerful_array(self, n: int) -> List[int]:
        """Generate the powerful array for a given integer n."""
        result = []
        i = 0
        while n > 0:
            if n & 1:
                result.append(2 ** i)
            n >>= 1
            i += 1
        return result[::-1]

    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        """Calculate the product of elements in big_nums for each query."""
        result = []
        for query in queries:
            from_i, to_i, mod_i = query
            product = 1
            for i in range(from_i, to_i + 1):
                powerful_array = self.powerful_array(i + 1)
                for num in powerful_array:
                    product = (product * num) % mod_i
            result.append(product)
        return result