class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def get_powerful_array(n):
            arr = []
            power = 1
            while n > 0:
                if n & 1:
                    arr.append(power)
                n >>= 1
                power <<= 1
            return arr

        def get_big_nums_index(index):
            count = 0
            num = 1
            while True:
                arr_len = len(get_powerful_array(num))
                if count + arr_len > index:
                    return get_powerful_array(num)[index - count]
                count += arr_len
                num += 1

        answer = []
        for query in queries:
            from_i, to_i, mod_i = query
            product = 1
            for i in range(from_i, to_i + 1):
                product = (product * get_big_nums_index(i)) % mod_i
            answer.append(product)
        return answer