# YOUR CODE HERE
S = list(map(int, input().split()))

# Check if the sequence is non-decreasing
is_non_decreasing = all(S[i] <= S[i+1] for i in range(len(S)-1))

# Check if all integers are between 100 and 675, inclusive
all_in_range = all(100 <= s <= 675 for s in S)

# Check if all integers are multiples of 25
all_multiples_of_25 = all(s % 25 == 0 for s in S)

if is_non_decreasing and all_in_range and all_multiples_of_25:
    print("Yes")
else:
    print("No")