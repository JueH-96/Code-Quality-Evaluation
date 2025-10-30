class Solution:
    def minimumCost(self, nums):
        def is_palindromic(num):
            return str(num) == str(num)[::-1]

        def generate_palindromic_numbers():
            palindromic_numbers = []
            for i in range(1, 1000000000):
                if is_palindromic(i):
                    palindromic_numbers.append(i)
            return palindromic_numbers

        def calculate_cost(nums, target):
            return sum(abs(num - target) for num in nums)

        palindromic_numbers = generate_palindromic_numbers()
        min_cost = float('inf')

        for palindromic_number in palindromic_numbers:
            cost = calculate_cost(nums, palindromic_number)
            min_cost = min(min_cost, cost)

        return min_cost

# Example usage:
# solution = Solution()
# print(solution.minimumCost([1, 2, 3, 4, 5]))  # Output: 6
# print(solution.minimumCost([10, 12, 13, 14, 15]))  # Output: 11
# print(solution.minimumCost([22, 33, 22, 33, 22]))  # Output: 22