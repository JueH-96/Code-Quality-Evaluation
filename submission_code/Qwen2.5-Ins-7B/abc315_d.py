# YOUR CODE HERE
H, W = map(int, input().split())
cookies = [input() for _ in range(H)]
rows = [[0] * 26 for _ in range(H)]
cols = [[0] * 26 for _ in range(W)]
marked = True

while marked:
    marked = False
    for i in range(H):
        count = 1
        color = ord(cookies[i][0]) - ord('a')
        for j in range(1, W):
            if cookies[i][j] == cookies[i][j-1]:
                count += 1
                if count >= 2:
                    rows[i][color] += 1
            else:
                count = 1
                color = ord(cookies[i][j]) - ord('a')
        if count >= 2:
            rows[i][color] += 1

    for j in range(W):
        count = 1
        color = ord(cookies[0][j]) - ord('a')
        for i in range(1, H):
            if cookies[i][j] == cookies[i-1][j]:
                count += 1
                if count >= 2:
                    cols[j][color] += 1
            else:
                count = 1
                color = ord(cookies[i][j]) - ord('a')
        if count >= 2:
            cols[j][color] += 1

    for i in range(H):
        for j in range(W):
            if rows[i][ord(cookies[i][j]) - ord('a')] >= 2 or cols[j][ord(cookies[i][j]) - ord('a')] >= 2:
                cookies[i] = cookies[i][:j] + '.' + cookies[i][j+1:]
                marked = True

print(sum(cookies[i].count('.') == 0 for i in range(H)) * W - sum(cookies[i] for i in range(H)).count('.'))