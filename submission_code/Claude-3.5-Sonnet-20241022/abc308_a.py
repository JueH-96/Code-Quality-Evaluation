S = list(map(int, input().split()))

# Check if sequence is monotonically non-decreasing
is_monotonic = all(S[i] <= S[i+1] for i in range(len(S)-1))

# Check if all numbers are between 100 and 675 inclusive
in_range = all(100 <= x <= 675 for x in S)

# Check if all numbers are multiples of 25
is_multiple_25 = all(x % 25 == 0 for x in S)

if is_monotonic and in_range and is_multiple_25:
    print("Yes")
else:
    print("No")