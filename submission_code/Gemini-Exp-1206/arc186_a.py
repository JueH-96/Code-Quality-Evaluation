def solve():
    n, q = map(int, input().split())
    queries = [int(input()) for _ in range(q)]

    possible_fixed_counts = set()
    possible_fixed_counts.add(0)
    possible_fixed_counts.add(n * n)

    if n % 2 == 0:
        for i in range(n // 2 + 1):
            for j in range(n // 2 + 1):
                possible_fixed_counts.add(2 * (i * (n - j) + j * (n - i) - 2 * i * j))

    
    for k in queries:
        if k in possible_fixed_counts:
            print("Yes")
        else:
            print("No")

solve()