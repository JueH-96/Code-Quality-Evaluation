# YOUR CODE HERE
def find_good_sequences(L, R):
    M = 0
    l, r = L, R
    while l < r:
        i = 0
        while (1 << i) * l < r:
            i += 1
        l = (1 << i) * l
        r = (1 << i) * l
        M += 1
    return M, l, r

L, R = map(int, input().split())
M, l, r = find_good_sequences(L, R)
print(M)
for _ in range(M):
    print(l, r)
    l = r
    r = R