# Read the input
S = list(map(int, input().split()))

# Check condition 1: monotonically non-decreasing
condition1 = all(S[i] <= S[i+1] for i in range(7))

# Check condition 2: all between 100 and 675 inclusive
condition2 = all(100 <= s <= 675 for s in S)

# Check condition 3: all multiples of 25
condition3 = all(s % 25 == 0 for s in S)

# Check if all conditions are satisfied
if condition1 and condition2 and condition3:
    print("Yes")
else:
    print("No")