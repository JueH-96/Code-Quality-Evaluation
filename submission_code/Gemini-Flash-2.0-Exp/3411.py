class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def get_powerful_array(x):
            powerful_array = []
            power = 1
            while x > 0:
                if x % 2 == 1:
                    powerful_array.append(power)
                x //= 2
                power *= 2
            return powerful_array

        def generate_big_nums(n):
            big_nums = []
            for i in range(1, n + 1):
                big_nums.extend(get_powerful_array(i))
            return big_nums

        def get_big_nums_element(index):
            num = 1
            count = 0
            while True:
                powerful_array = get_powerful_array(num)
                if count + len(powerful_array) > index:
                    return powerful_array[index - count]
                count += len(powerful_array)
                num += 1

        result = []
        for query in queries:
            from_i, to_i, mod_i = query
            product = 1
            for i in range(from_i, to_i + 1):
                product = (product * get_big_nums_element(i)) % mod_i
            result.append(product)
        return result