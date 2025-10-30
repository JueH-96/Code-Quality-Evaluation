class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        result = []
        for query in queries:
            from_i, to_i, mod_i = query
            product = 1
            for i in range(from_i, to_i + 1):
                product = (product * big_nums[i]) % mod_i
            result.append(product)
        return result