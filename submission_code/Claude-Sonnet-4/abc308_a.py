# YOUR CODE HERE
# Read input
numbers = list(map(int, input().split()))

# Check condition 1: monotonically non-decreasing
is_non_decreasing = True
for i in range(1, 8):
    if numbers[i] < numbers[i-1]:
        is_non_decreasing = False
        break

# Check condition 2: all between 100 and 675 inclusive
in_range = all(100 <= num <= 675 for num in numbers)

# Check condition 3: all multiples of 25
multiples_of_25 = all(num % 25 == 0 for num in numbers)

# Print result
if is_non_decreasing and in_range and multiples_of_25:
    print("Yes")
else:
    print("No")