import sys

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    N = int(lines[0].strip())
    A = list(map(int, lines[1].strip().split()))
    return N, A

def calculate_f(x, y):
    """Calculate f(x, y) as per the problem definition."""
    z = int(str(x) + str(y))
    return z

def calculate_sum(N, A):
    """Calculate the sum of f(A_i, A_j) for all pairs (i, j) where i < j."""
    total_sum = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            total_sum += calculate_f(A[i], A[j])
    return total_sum % 998244353

def main():
    N, A = read_input()
    result = calculate_sum(N, A)
    print(result)

if __name__ == "__main__":
    main()