s = list(map(int, input().split()))

# Check each number is a multiple of 25 and within the range 100-675
for num in s:
    if num % 25 != 0:
        print("No")
        exit()
    if num < 100 or num > 675:
        print("No")
        exit()

# Check if the sequence is non-decreasing
for i in range(7):
    if s[i] > s[i + 1]:
        print("No")
        exit()

print("Yes")