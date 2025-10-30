# Read the sequence of integers from standard input
S = list(map(int, input().split()))

# Check if the sequence is monotonically non-decreasing
is_non_decreasing = all(S[i] <= S[i+1] for i in range(len(S)-1))

# Check if all numbers are between 100 and 675, inclusive
is_in_range = all(100 <= num <= 675 for num in S)

# Check if all numbers are multiples of 25
is_multiple_of_25 = all(num % 25 == 0 for num in S)

# Print Yes if all conditions are satisfied, otherwise print No
if is_non_decreasing and is_in_range and is_multiple_of_25:
    print("Yes")
else:
    print("No")