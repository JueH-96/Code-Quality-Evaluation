# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
L = int(data[1])
A = list(map(int, data[2:]))

count = sum(1 for score in A if score >= L)
print(count)