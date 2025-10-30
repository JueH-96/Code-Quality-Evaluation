# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
B = int(data[2])
D = list(map(int, data[3:]))

# Check if all D_i % (A+B) are within the holiday range
for d in D:
    if d % (A + B) > A:
        print("No")
        break
else:
    print("Yes")