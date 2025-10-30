import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])

    # The tree has N*M + 1 vertices
    total_vertices = N * M + 1

    # Expected number of operations
    expected_operations = 2 * N * M * (N + 1)

    # Output the result modulo 998244353
    print(expected_operations % MOD)

if __name__ == "__main__":
    main()