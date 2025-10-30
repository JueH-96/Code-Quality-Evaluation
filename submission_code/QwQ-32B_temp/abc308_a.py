import sys

s = list(map(int, input().split()))

# Check if all numbers are between 100 and 675 inclusive
for num in s:
    if num < 100 or num > 675:
        print("No")
        sys.exit()

# Check if all numbers are multiples of 25
for num in s:
    if num % 25 != 0:
        print("No")
        sys.exit()

# Check if the sequence is non-decreasing
for i in range(7):
    if s[i] > s[i+1]:
        print("No")
        sys.exit()

print("Yes")