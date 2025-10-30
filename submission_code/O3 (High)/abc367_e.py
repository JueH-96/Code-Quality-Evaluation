import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    # No operation to perform
    if K == 0:
        print(' '.join(map(str, A)))
        return

    # 0-index the permutation
    power = [x - 1 for x in X]      # current 1-step jump list: f(i)
    res   = list(range(N))          # res[i] stores f^applied(i)

    while K:
        if K & 1:                   # apply current power if the bit is set
            res = [power[i] for i in res]

        K >>= 1
        if K == 0:                  # no more bits, done
            break

        # square the permutation: power = power ∘ power
        power = [power[i] for i in power]

    # build the final sequence
    answer = [A[i] for i in res]
    print(' '.join(map(str, answer)))

if __name__ == "__main__":
    main()