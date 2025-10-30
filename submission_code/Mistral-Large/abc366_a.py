import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
A = int(data[2])

remaining_votes = N - (T + A)

if T > A and T > A + remaining_votes:
    print("Yes")
elif A > T and A > T + remaining_votes:
    print("Yes")
else:
    print("No")