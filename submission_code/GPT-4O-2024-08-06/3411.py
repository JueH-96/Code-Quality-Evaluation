class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        import heapq
        from typing import List

        # Function to generate the powerful array for a given number x
        def powerful_array(x):
            powers = []
            power = 1
            while x > 0:
                if x % 2 == 1:
                    powers.append(power)
                x //= 2
                power *= 2
            return powers

        # Function to generate the big_nums array up to a certain length
        def generate_big_nums(length):
            big_nums = []
            i = 1
            while len(big_nums) < length:
                big_nums.extend(powerful_array(i))
                i += 1
            return big_nums

        # Precompute the big_nums array up to a reasonable length
        # Since the queries can go up to 10^15, we need a strategy to handle this
        # We will generate a large enough portion of big_nums to handle small queries efficiently
        big_nums = generate_big_nums(1000)  # Generate a reasonable length for small queries

        # Function to handle a single query
        def handle_query(from_i, to_i, mod_i):
            # If the query range is within the precomputed big_nums, use it
            if to_i < len(big_nums):
                product = 1
                for i in range(from_i, to_i + 1):
                    product = (product * big_nums[i]) % mod_i
                return product
            else:
                # If the query range is beyond precomputed big_nums, generate on-the-fly
                product = 1
                current_index = 0
                current_number = 1
                while current_index <= to_i:
                    powers = powerful_array(current_number)
                    for power in powers:
                        if current_index >= from_i:
                            product = (product * power) % mod_i
                        current_index += 1
                        if current_index > to_i:
                            break
                    current_number += 1
                return product

        # Process each query
        answers = []
        for query in queries:
            from_i, to_i, mod_i = query
            result = handle_query(from_i, to_i, mod_i)
            answers.append(result)

        return answers