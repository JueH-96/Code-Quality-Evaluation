class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # Initialize the list big_nums
        big_nums = [1]
        # Initialize the power of two
        power = 2
        # While the last element in big_nums is less than 10^15
        while big_nums[-1] < 10**15:
            # Append the power of two to big_nums
            big_nums.append(power)
            # Double the power of two
            power *= 2
        # Initialize the result list
        result = []
        # For each query
        for query in queries:
            # Calculate the product of the elements in big_nums from query[0] to query[1]
            product = 1
            for num in big_nums[query[0]:query[1]+1]:
                product = (product * num) % query[2]
            # Append the product to the result list
            result.append(product)
        # Return the result list
        return result