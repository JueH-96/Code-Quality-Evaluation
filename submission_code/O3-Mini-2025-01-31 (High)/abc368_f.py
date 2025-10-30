def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    max_val = max(A)
    
    # Precompute proper divisors for each number from 1 to max_val.
    # For each x, divisors[x] will contain all d such that d divides x and d < x.
    divisors = [[] for _ in range(max_val + 1)]
    for d in range(1, max_val + 1):
        for m in range(2 * d, max_val + 1, d):
            divisors[m].append(d)
    
    # Compute Grundy numbers for each value.
    # Terminal position: 1 has no proper move so grundy[1] = 0.
    grundy = [0] * (max_val + 1)
    for x in range(2, max_val + 1):
        moves = set()
        for d in divisors[x]:
            moves.add(grundy[d])
        g = 0
        while g in moves:
            g += 1
        grundy[x] = g
    
    # The overall game is the disjunctive sum of the games on each number.
    # Its outcome is determined by the XOR (nim-sum) of the Grundy numbers.
    nim_sum = 0
    for a in A:
        nim_sum ^= grundy[a]
    
    if nim_sum != 0:
        sys.stdout.write("Anna")
    else:
        sys.stdout.write("Bruno")

if __name__ == '__main__':
    main()