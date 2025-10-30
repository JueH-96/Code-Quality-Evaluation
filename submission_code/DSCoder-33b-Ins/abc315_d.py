# YOUR CODE HERE

H, W = map(int, input().split())
cookies = [list(input()) for _ in range(H)]

while True:
    marked = [[False]*W for _ in range(H)]
    for i in range(H):
        if len(set(cookies[i])) == 1:
            marked[i] = [True]*W
    for j in range(W):
        if len(set(cookies[i][j] for i in range(H))) == 1:
            for i in range(H):
                marked[i][j] = True
    if any(any(row) for row in marked):
        cookies = [[cookies[i][j] for j in range(W) if not marked[i][j]] for i in range(H)]
        H, W = len(cookies), len(cookies[0])
    else:
        break

print(sum(len(row) for row in cookies))