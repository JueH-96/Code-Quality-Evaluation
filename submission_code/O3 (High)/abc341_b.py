import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    idx = 0

    N = data[idx]; idx += 1
    A = data[idx: idx + N]; idx += N

    S = [0] * (N - 1)
    T = [0] * (N - 1)
    for i in range(N - 1):
        S[i] = data[idx]; idx += 1
        T[i] = data[idx]; idx += 1

    # amount currently held in the currency of country i (0-indexed)
    amount = A[0]

    for i in range(N - 1):
        k = amount // S[i]         # number of times we can execute the i-th exchange
        amount = A[i + 1] + k * T[i]   # what reaches country (i+1)

    print(amount)

if __name__ == "__main__":
    main()