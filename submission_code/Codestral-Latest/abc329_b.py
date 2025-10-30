# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

max_val = max(A)
second_max = max(x for x in A if x != max_val)

print(second_max)