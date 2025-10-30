import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

for i in range(1, N):
    if (S[i-1] == 'a' and S[i] == 'b') or (S[i-1] == 'b' and S[i] == 'a'):
        print("Yes")
        sys.exit(0)

print("No")