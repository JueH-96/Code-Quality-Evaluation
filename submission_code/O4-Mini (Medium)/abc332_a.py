def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    S = int(next(it))
    K = int(next(it))
    total = 0
    for _ in range(n):
        p = int(next(it))
        q = int(next(it))
        total += p * q
    if total < S:
        total += K
    print(total)

main()