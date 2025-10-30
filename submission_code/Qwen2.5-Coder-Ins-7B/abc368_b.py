# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

count = 0
while len([x for x in A if x > 0]) > 1:
    A.sort(reverse=True)
    A[0] -= 1
    A[1] -= 1
    count += 1

print(count)