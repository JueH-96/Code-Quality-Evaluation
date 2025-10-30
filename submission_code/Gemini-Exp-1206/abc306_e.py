def solve():
    n, k, q = map(int, input().split())
    a = [0] * n
    
    def calculate_f(arr, k_val):
        b = sorted(arr, reverse=True)
        return sum(b[:k_val])

    for _ in range(q):
        x, y = map(int, input().split())
        a[x - 1] = y
        print(calculate_f(a, k))

solve()