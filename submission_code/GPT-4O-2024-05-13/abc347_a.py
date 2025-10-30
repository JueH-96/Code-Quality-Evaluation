# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

result = [a // K for a in A if a % K == 0]
print(" ".join(map(str, sorted(result))))