import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]

    # Special case when M == 2:
    # with only two symbols, the only good sequences are strict alternations.
    # You cannot change a single element without matching a neighbor.
    # So if A != B, it's impossible; otherwise zero moves.
    if M == 2:
        if A == B:
            print(0)
        else:
            print(-1)
        return

    # For M >= 3, it's always possible to adjust each position step by step
    # without ever forcing an adjacency-equal state, because you can "detour"
    # through a third value if needed.
    # The minimum number of +1/-1 moves mod M to go from a to b is:
    #   min((b-a) mod M, (a-b) mod M)
    total = 0
    for a, b in zip(A, B):
        diff = abs(a - b)
        total += min(diff, M - diff)
    print(total)

if __name__ == "__main__":
    main()