n, m = map(int, input().split())
s = [list(input()) for _ in range(n)]
ans = []
for i in range(n-8):
    for j in range(m-8):
        flag = 1
        for k in range(3):
            for l in range(3):
                if s[i+k][j+l] != "#" or s[i+k+6][j+l] != "#":
                    flag = 0
        for k in range(1, 8):
            for l in range(3):
                if s[i+k][j+l] != "." or s[i+k][j+l+6] != ".":
                    flag = 0
        for k in range(3):
            for l in range(1, 8):
                if s[i+k][j+l] != "." or s[i+k+6][j+l] != ".":
                    flag = 0
        if flag:
            ans.append([i+1, j+1])
for a in ans:
    print(a[0], a[1])