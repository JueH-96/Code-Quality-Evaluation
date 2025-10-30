# YOUR CODE HERE
# Read the input
S = list(map(int, input().split()))

# Check condition 1: monotonically non-decreasing
is_non_decreasing = True
for i in range(1, 8):
    if S[i] < S[i-1]:
        is_non_decreasing = False
        break

# Check condition 2: all values between 100 and 675
all_in_range = True
for s in S:
    if s < 100 or s > 675:
        all_in_range = False
        break

# Check condition 3: all values are multiples of 25
all_multiples_of_25 = True
for s in S:
    if s % 25 != 0:
        all_multiples_of_25 = False
        break

# Print result
if is_non_decreasing and all_in_range and all_multiples_of_25:
    print("Yes")
else:
    print("No")