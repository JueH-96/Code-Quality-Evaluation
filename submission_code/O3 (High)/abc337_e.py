import sys

def main() -> None:
    # Read the number of bottles
    n_line = ''
    while n_line.strip() == '':
        n_line = sys.stdin.readline()
        if n_line == '':
            return                      # no input
    N = int(n_line.strip())

    # Minimum number of friends (bits) needed
    M = 0
    while (1 << M) < N:
        M += 1

    # Output M
    print(M, flush=True)

    # For each bit position i (friend i+1), give them
    # all bottles whose (i-th) bit of (index-1) is 1.
    for i in range(M):
        bottles = [b for b in range(1, N + 1) if ((b - 1) >> i) & 1]
        print(len(bottles), *bottles, flush=True)

    # Read the results string S (length M, characters ‘0’/‘1’)
    S = ''
    while S.strip() == '':
        S = sys.stdin.readline()
        if S == '':
            return                      # no more input
    S = S.strip()

    # Decode the spoiled bottle
    x = 0
    for i, ch in enumerate(S):
        if ch == '1':
            x |= 1 << i
    spoiled = x + 1                    # bottle numbers are 1-based

    # Output the answer
    print(spoiled, flush=True)

if __name__ == '__main__':
    main()