class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        
        def get_powerful_array(x):
            powerful_array = []
            power = 1
            while x > 0:
                if x & 1:
                    powerful_array.append(power)
                x >>= 1
                power <<= 1
            return powerful_array
        
        def generate_big_nums(limit):
            big_nums = []
            i = 1
            while len(big_nums) <= limit:
                big_nums.extend(get_powerful_array(i))
                i += 1
            return big_nums
        
        max_to = 0
        for query in queries:
            max_to = max(max_to, query[1])
        
        big_nums = generate_big_nums(max_to)
        
        results = []
        for query in queries:
            from_i, to_i, mod_i = query
            product = 1
            for i in range(from_i, to_i + 1):
                product = (product * big_nums[i]) % mod_i
            results.append(product)
        
        return results