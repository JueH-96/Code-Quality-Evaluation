import sys

def main() -> None:
    """
    Offline solution for the adapted interactive task.

    Input format (adapted):
        N L R
        A_0 A_1 ... A_{2^N-1}

    Output:
        (A_L + ... + A_R) mod 100
    """
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return

    # Extract N, L, R
    N, L, R = data[0], data[1], data[2]

    # Remaining numbers correspond to the array A
    A = data[3:]

    # Compute and output the required remainder
    ans = sum(A[L:R + 1]) % 100
    print(ans)

if __name__ == "__main__":
    main()