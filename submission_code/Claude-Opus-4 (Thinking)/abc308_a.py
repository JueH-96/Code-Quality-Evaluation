# Read input
s = list(map(int, input().split()))

# Check all conditions
cond1 = all(s[i] <= s[i+1] for i in range(7))
cond2 = all(100 <= val <= 675 for val in s)
cond3 = all(val % 25 == 0 for val in s)

# Print result
if cond1 and cond2 and cond3:
    print("Yes")
else:
    print("No")