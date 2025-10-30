# YOUR CODE HERE
H, W = map(int, input().split())
cookies = [list(input()) for _ in range(H)]

def mark_cookies(h, w, cookies):
    for i in range(h):
        row = cookies[i]
        if len(set(row)) > 1:
            for j in range(w):
                if row[j] != row[0]:
                    row[j] = '.'
    for j in range(w):
        col = [cookies[i][j] for i in range(h)]
        if len(set(col)) > 1:
            for i in range(h):
                if col[i] != col[0]:
                    cookies[i][j] = '.'

mark_cookies(H, W, cookies)

remaining_cookies = sum(row.count('.') for row in cookies)
print(remaining_cookies)