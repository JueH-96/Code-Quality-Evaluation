# Read the input from stdin
input_values = [int(x) for x in input().split()]

# Check if the sequence is monotonically non-decreasing
is_monotonic = all(input_values[i] <= input_values[i+1] for i in range(len(input_values)-1))

# Check if all values are between 100 and 675, inclusive
is_in_range = all(100 <= x <= 675 for x in input_values)

# Check if all values are multiples of 25
is_multiple_of_25 = all(x % 25 == 0 for x in input_values)

# Print the answer
if is_monotonic and is_in_range and is_multiple_of_25:
    print("Yes")
else:
    print("No")