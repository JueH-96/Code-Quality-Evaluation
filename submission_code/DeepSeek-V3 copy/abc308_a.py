# Read the input as a list of integers
S = list(map(int, input().split()))

# Check if the sequence is monotonically non-decreasing
non_decreasing = all(S[i] <= S[i+1] for i in range(len(S)-1))

# Check if all elements are between 100 and 675, inclusive
in_range = all(100 <= s <= 675 for s in S)

# Check if all elements are multiples of 25
multiples_of_25 = all(s % 25 == 0 for s in S)

# If all conditions are satisfied, print Yes; otherwise, print No
if non_decreasing and in_range and multiples_of_25:
    print("Yes")
else:
    print("No")