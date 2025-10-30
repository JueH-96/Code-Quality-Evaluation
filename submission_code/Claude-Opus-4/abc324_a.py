# YOUR CODE HERE
n = int(input())
numbers = list(map(int, input().split()))

# Check if all numbers are equal to the first number
if all(num == numbers[0] for num in numbers):
    print("Yes")
else:
    print("No")