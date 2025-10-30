class Solution:
    def minimumOperations(self, num: str) -> int:
        # Initialize the count of operations
        operations = 0
        # While the number is not divisible by 25
        while int(num) % 25 != 0:
            # Find the smallest digit that is greater than the first digit
            smallest_greater = min(filter(lambda x: x > num[0], num))
            # Delete all digits that are less than the smallest greater digit
            num = num.replace(min(filter(lambda x: x < smallest_greater, num)), '')
            # Increment the count of operations
            operations += 1
        # Return the count of operations
        return operations