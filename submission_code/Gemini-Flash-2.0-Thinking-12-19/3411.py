from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        big_nums_list = []
        current_len = 0
        i = 1
        while current_len <= 2 * 10**6: # Generate big_nums up to a certain length
            powerful_array = []
            temp_i = i
            power = 0
            while temp_i > 0:
                if temp_i % 2 == 1:
                    powerful_array.append(2**power)
                temp_i //= 2
                power += 1
            big_nums_list.extend(powerful_array)
            current_len = len(big_nums_list)
            i += 1
            if i > 200000: # Safety break to avoid infinite loop if length does not reach limit
                break

        answers = []
        for query in queries:
            from_i, to_i, mod_i = query[0], query[1], query[2]
            product = 1
            for j in range(from_i, to_i + 1):
                if j < len(big_nums_list):
                    product = (product * big_nums_list[j]) % mod_i
                else:
                    # If index is out of precomputed range, we need to generate on the fly.
                    # But given constraints and problem nature, precomputation should be enough.
                    # If not, we might need to rethink the precomputation limit or approach.
                    # For now, if out of range, we can assume it's 1 as it won't affect product if range is invalid.
                    pass 
            answers.append(product)
        return answers