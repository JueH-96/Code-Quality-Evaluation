def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    bottom_k = a[n-k:]
    top_n_minus_k = a[:n-k]

    b = bottom_k + top_n_minus_k
    print(*b)

if __name__ == "__main__":
    solve()