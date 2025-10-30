import sys
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]
for i in range(N-2):
    if S[i:i+3] == "ABC":
        print(i+1)
        break
else:
    print(-1)