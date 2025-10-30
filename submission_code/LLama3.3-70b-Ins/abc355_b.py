import sys

def solve():
    # Read input from stdin
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Combine and sort the sequences
    C = sorted(A + B)

    # Check for consecutive elements from A
    for i in range(len(C) - 1):
        if C[i] in A and C[i + 1] in A:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    solve()