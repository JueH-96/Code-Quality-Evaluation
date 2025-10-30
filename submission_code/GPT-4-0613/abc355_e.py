def query(i, j):
    print(f'? {i} {j}')
    return int(input())

def solve(l, r):
    if l == r:
        return query(0, l)
    m = (l + r) // 2
    return (solve(l, m) + solve(m + 1, r)) % 100

N, L, R = map(int, input().split())
print(f'! {solve(L, R)}')