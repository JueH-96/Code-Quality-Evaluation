import sys

input = sys.stdin.read
data = input().strip()

# Check if the i-th character of S is 0 for every even number i from 2 through 16
for i in range(2, 17, 2):
    if data[i-1] != '0':
        print("No")
        break
else:
    print("Yes")