def solve():
    n = int(input())
    if n == 1:
        print(1)
        return

    results = set()
    results.add(1)  # Handle the case of 1

    max_b = 60  # A safe upper bound for the exponent

    for b in range(2, max_b + 1):
        max_a = int(n ** (1 / b))
        for a in range(2, max_a + 2):
            try:
                x = a ** b
                if x <= n:
                    results.add(x)
                else:
                    break
            except OverflowError:
                break

    print(len(results))

solve()