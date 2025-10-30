import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

min_passengers = min(0, sum(A))

print(min_passengers)