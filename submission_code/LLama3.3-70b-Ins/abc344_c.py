import sys

def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    M = int(input())
    B = list(map(int, input().split()))
    L = int(input())
    C = list(map(int, input().split()))
    Q = int(input())
    X = list(map(int, input().split()))

    # Generate all possible sums
    sums = set()
    for a in A:
        for b in B:
            for c in C:
                sums.add(a + b + c)

    # Check each query
    for x in X:
        if x in sums:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    solve()