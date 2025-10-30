# YOUR CODE HERE
N, M, H, K = map(int, input().split())
S = input()
items = {}
for i in range(M):
    x, y = map(int, input().split())
    items[(x, y)] = True

x, y = 0, 0
for move in S:
    if move == 'R':
        x += 1
    elif move == 'L':
        x -= 1
    elif move == 'U':
        y += 1
    elif move == 'D':
        y -= 1
    H -= 1
    if H < 0:
        print('No')
        break
    if (x, y) in items and H < K:
        H = K
else:
    print('Yes')