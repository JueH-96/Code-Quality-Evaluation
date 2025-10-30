# YOUR CODE HERE
M, D = map(int, input().split())
y, m, d = map(int, input().split())

if d < D:
    y_new, m_new, d_new = y, m, d + 1
else:
    if m < M:
        y_new, m_new, d_new = y, m + 1, 1
    else:
        y_new, m_new, d_new = y + 1, 1, 1

print(y_new, m_new, d_new)