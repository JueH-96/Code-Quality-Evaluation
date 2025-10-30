class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # Create the big_nums array
        big_nums = []
        for i in range(1, 10**15+1):
            powers_of_two = []
            x = i
            while x > 0:
                if x % 2 == 1:
                    powers_of_two.append(1)
                else:
                    powers_of_two.append(x)
                    break
                x //= 2
            big_nums.extend(powers_of_two)
        
        # Process the queries
        answer = []
        for from_i, to_i, mod_i in queries:
            product = 1
            for j in range(from_i-1, to_i):
                product *= big_nums[j]
            answer.append(product % mod_i)
        
        return answer