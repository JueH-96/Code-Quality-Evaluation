# YOUR CODE HERE

# Read the inputs
inputs = list(map(int, input().split()))

# Check if the sequence is monotonically non-decreasing
is_non_decreasing = all(inputs[i] <= inputs[i+1] for i in range(len(inputs)-1))

# Check if all numbers are between 100 and 675 inclusive
is_between_100_675 = all(100 <= num <= 675 for num in inputs)

# Check if all numbers are multiples of 25
is_multiple_of_25 = all(num % 25 == 0 for num in inputs)

# Print the result
if is_non_decreasing and is_between_100_675 and is_multiple_of_25:
    print("Yes")
else:
    print("No")