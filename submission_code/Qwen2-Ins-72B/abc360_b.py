S, T = input().split()

for c in range(1, len(S)):
    for w in range(c, len(S)):
        if w >= len(S):
            break
        ans = ""
        for i in range(0, len(S), w):
            if i + c <= len(S):
                ans += S[i + c - 1]
        if ans == T:
            print("Yes")
            exit()
print("No")