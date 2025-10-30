import sys
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]
if any((S[i] == 'a' and S[i + 1] == 'b') or (S[i] == 'b' and S[i + 1] == 'a') for i in range(N - 1)):
    print("Yes")
else:
    print("No")