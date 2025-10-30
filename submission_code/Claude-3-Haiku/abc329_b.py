# Read the input
n = int(input())
nums = list(map(int, input().split()))

# Find the largest number
largest = max(nums)

# Find the largest number that is not the largest
second_largest = max(num for num in nums if num != largest)

# Print the answer
print(second_largest)