def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    max_A = max(A)
    
    # Build a list of divisors for every number from 1 to max_A.
    # We include all divisors, then later ignore the number itself.
    divisors = [[] for _ in range(max_A + 1)]
    for i in range(1, max_A + 1):
        for j in range(i, max_A + 1, i):
            divisors[j].append(i)

    # Grundy number for game state "number" where possible moves are to any proper divisor.
    # Terminal state: 1 has no proper divisors other than itself, so g(1) = 0.
    grundy = [0] * (max_A + 1)
    grundy[1] = 0
    for x in range(2, max_A + 1):
        reachable = set()
        for d in divisors[x]:
            if d == x:
                continue  # d must be a proper divisor.
            reachable.add(grundy[d])
        g = 0
        while g in reachable:
            g += 1
        grundy[x] = g

    # Compute overall nim-sum. The game is a disjunctive sum, so Anna wins if nim-sum != 0.
    nim_sum = 0
    for value in A:
        nim_sum ^= grundy[value]

    if nim_sum != 0:
        sys.stdout.write("Anna")
    else:
        sys.stdout.write("Bruno")

if __name__ == '__main__':
    main()