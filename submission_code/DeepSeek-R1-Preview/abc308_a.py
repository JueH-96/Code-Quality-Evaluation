# Read the input as a list of integers
s = list(map(int, input().split()))

# Check condition 1: The sequence is monotonically non-decreasing
for i in range(7):
    if s[i] > s[i+1]:
        print("No")
        exit()

# Check condition 2: All elements are between 100 and 675, inclusive
for num in s:
    if not (100 <= num <= 675):
        print("No")
        exit()

# Check condition 3: All elements are multiples of 25
for num in s:
    if num % 25 != 0:
        print("No")
        exit()

# All conditions are satisfied
print("Yes")