# Read the input as a list of integers
s = list(map(int, input().split()))

# Check the first condition: monotonically non-decreasing
cond1 = True
for i in range(7):
    if s[i] > s[i+1]:
        cond1 = False
        break

# Check the second condition: all between 100 and 675 inclusive
cond2 = all(100 <= x <= 675 for x in s)

# Check the third condition: all multiples of 25
cond3 = all(x % 25 == 0 for x in s)

# Determine and print the result
if cond1 and cond2 and cond3:
    print("Yes")
else:
    print("No")