def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    A = list(map(int, input().split()))
    S = sum(A)
    q, r = divmod(S, N)

    # We want exactly r elements to become q+1, the rest become q.
    # Assign the largest r elements to target q+1 to minimize moves.
    A.sort(reverse=True)
    ops = 0
    # For the top r elements, their target is q+1
    # For the remaining N-r elements, target is q
    for i, a in enumerate(A):
        target = q + 1 if i < r else q
        if a > target:
            ops += a - target

    print(ops)

if __name__ == "__main__":
    main()