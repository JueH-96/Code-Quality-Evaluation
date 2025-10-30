import sys

N = int(sys.stdin.readline())
S = [sys.stdin.readline().strip() for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i != j:
            concatenated = S[i] + S[j]
            if concatenated == concatenated[::-1]:
                print("Yes")
                sys.exit()
print("No")