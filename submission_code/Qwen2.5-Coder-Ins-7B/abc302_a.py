# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
A = int(data[0])
B = int(data[1])

# Calculate the minimum number of attacks needed
# If A is not divisible by B, we need an additional attack to reduce the stamina to 0 or less
if A % B == 0:
    print(A // B)
else:
    print(A // B + 1)