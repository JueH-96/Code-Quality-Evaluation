import sys

def main():
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # Any two-term sequence is always a geometric progression
    if n <= 2:
        print("Yes")
        return

    # Let the common ratio be r = A[1] / A[0].
    # To avoid floating point errors, we check cross-multiplication:
    # A[i] * A[0] == A[i-1] * A[1] for all i >= 2
    num = A[1]
    den = A[0]
    for i in range(2, n):
        if A[i] * den != A[i-1] * num:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    main()