S = [input() for _ in range(8)]
ans = 0

for i in range(8):
    for j in range(8):
        if S[i][j] == "#":
            continue

        flag = True
        for k in range(8):
            if S[i][k] == "#" or S[k][j] == "#":
                flag = False
                break
        if flag:
            ans += 1

print(ans)