import sys

input = sys.stdin.readline

S, T = input().split()

for w in range(1, len(S)):
    ans = ""
    for i in range(0, len(S), w+1):
        if i+w > len(S):
            break
        ans += S[i+w-1]
    if ans == T:
        print("Yes")
        exit()

print("No")