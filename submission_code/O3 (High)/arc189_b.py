import sys

def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    N = int(input_data[0])
    X = list(map(int, input_data[1:]))

    # gaps d_j = X_{j+1} - X_{j}  (1-based j = 1..N-1)
    odd_gaps  = []   # j is odd   (1,3,5,…)
    even_gaps = []   # j is even  (2,4,6,…)

    for j in range(N - 1):        # j : 0-based
        g = X[j + 1] - X[j]
        if (j + 1) & 1:           # 1-based index is odd
            odd_gaps.append(g)
        else:
            even_gaps.append(g)

    odd_gaps.sort()
    even_gaps.sort()

    # rebuild gaps: within each parity we place smaller gaps first
    oi = ei = 0
    total = X[0] * N              # contribution of the left end-point

    for j in range(N - 1):        # 0-based position of gap
        if (j + 1) & 1:           # odd gap position
            d = odd_gaps[oi]
            oi += 1
        else:                     # even gap position
            d = even_gaps[ei]
            ei += 1
        weight = N - j - 1        # how many coordinates include this gap
        total += weight * d

    print(total)

if __name__ == "__main__":
    main()