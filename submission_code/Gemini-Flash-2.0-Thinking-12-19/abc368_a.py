def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    bottom_k_cards = a[n-k:]
    remaining_cards = a[:n-k]

    b = bottom_k_cards + remaining_cards
    print(*(b))

solve()