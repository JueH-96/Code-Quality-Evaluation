class Solution:
    def findProductsOfElements(self, queries):
        def get_powerful_array(n):
            result = []
            while n > 0:
                result.append(2 ** (n.bit_length() - 1))
                n -= result[-1]
            return result[::-1]

        def get_big_nums(n):
            result = []
            i = 1
            while len(result) < n:
                result.extend(get_powerful_array(i))
                i += 1
            return result[:n]

        def get_product(nums, mod):
            result = 1
            for num in nums:
                result = (result * num) % mod
            return result

        answer = []
        for query in queries:
            from_i, to_i, mod_i = query
            big_nums = get_big_nums(to_i + 1)
            answer.append(get_product(big_nums[from_i:to_i + 1], mod_i))
        return answer