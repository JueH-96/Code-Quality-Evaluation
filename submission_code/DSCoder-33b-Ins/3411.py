class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        big_nums = [1, 2]
        for i in range(3, 10**15 + 1):
            if i & (i-1) == 0:
                big_nums.append(i)
            else:
                big_nums.append(big_nums[i // 2] + big_nums[i // 2 + 1])

        result = []
        for query in queries:
            product = 1
            for i in range(query[0], query[1] + 1):
                product = (product * big_nums[i]) % query[2]
            result.append(product)

        return result