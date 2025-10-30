# Read the input and split into a list of integers
numbers = list(map(int, input().split()))

# Check if the sequence is monotonically non-decreasing
is_non_decreasing = all(numbers[i] <= numbers[i+1] for i in range(7))

# Check if all numbers are between 100 and 675 inclusive
are_in_range = all(100 <= num <= 675 for num in numbers)

# Check if all numbers are multiples of 25
are_multiples_of_25 = all(num % 25 == 0 for num in numbers)

# Determine the final result
if is_non_decreasing and are_in_range and are_multiples_of_25:
    print("Yes")
else:
    print("No")