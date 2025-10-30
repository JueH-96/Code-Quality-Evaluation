# YOUR CODE HERE
n = int(input())
numbers = list(map(int, input().split()))

# Check if all numbers are equal by comparing each with the first one
all_equal = all(num == numbers[0] for num in numbers)

if all_equal:
    print("Yes")
else:
    print("No")