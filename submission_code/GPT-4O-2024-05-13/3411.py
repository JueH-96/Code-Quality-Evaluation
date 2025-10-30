class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def powerful_array(x):
            powers = []
            power = 1
            while x > 0:
                if x % 2 == 1:
                    powers.append(power)
                x //= 2
                power *= 2
            return powers
        
        def generate_big_nums(limit):
            big_nums = []
            for i in range(1, limit + 1):
                big_nums.extend(powerful_array(i))
            return big_nums
        
        # Determine the maximum range we need to consider
        max_index = max(query[1] for query in queries)
        
        # Generate big_nums up to the required index
        big_nums = generate_big_nums(10**6)  # Generate a large enough big_nums array
        
        results = []
        for from_i, to_i, mod_i in queries:
            product = 1
            for i in range(from_i, to_i + 1):
                product = (product * big_nums[i]) % mod_i
            results.append(product)
        
        return results