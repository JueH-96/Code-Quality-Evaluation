class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        big_nums = []
        for i in range(1, 100001):
            temp = bin(i)[2:]
            for j in range(len(temp)):
                if temp[j] == '1':
                    big_nums.append(2**(len(temp) - 1 - j))

        result = []
        for query in queries:
            start, end, mod = query
            product = 1
            for i in range(start, end + 1):
                product = (product * big_nums[i]) % mod
            result.append(product)
        return result