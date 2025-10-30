def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))

    for _ in range(q):
        b_k = list(map(int, input().split()))
        b = b_k[0]
        k = b_k[1]

        distances = []
        for val_a in a:
            distances.append(abs(val_a - b))

        distances.sort()
        print(distances[k - 1])

solve()