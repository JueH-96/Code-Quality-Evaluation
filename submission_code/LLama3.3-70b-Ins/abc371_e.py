import sys

def calculate_f(l, r, A):
    """Calculate the number of distinct values in the subsequence (A_l, A_{l+1}, ..., A_r)"""
    return len(set(A[l-1:r]))

def main():
    # Read input from stdin
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    total_sum = 0
    for i in range(1, N+1):
        for j in range(i, N+1):
            total_sum += calculate_f(i, j, A)

    print(total_sum)

if __name__ == "__main__":
    main()