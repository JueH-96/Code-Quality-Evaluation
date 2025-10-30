import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

quotients = [a // K for a in A if a % K == 0]
quotients.sort()

print(" ".join(map(str, quotients)))