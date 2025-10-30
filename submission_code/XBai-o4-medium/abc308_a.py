# Read the input as a list of integers
s = list(map(int, input().split()))

# Check if the sequence is non-decreasing
cond1 = True
for i in range(7):
    if s[i] > s[i+1]:
        cond1 = False
        break

# Check if all elements are between 100 and 675 inclusive
cond2 = True
for num in s:
    if not (100 <= num <= 675):
        cond2 = False
        break

# Check if all elements are multiples of 25
cond3 = True
for num in s:
    if num % 25 != 0:
        cond3 = False
        break

# Output the result based on all conditions
if cond1 and cond2 and cond3:
    print("Yes")
else:
    print("No")