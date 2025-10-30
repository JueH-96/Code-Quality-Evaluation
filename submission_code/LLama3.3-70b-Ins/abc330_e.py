import sys

def mex(A):
    """Calculate the mex of a list A"""
    A = set(A)
    mex = 0
    while mex in A:
        mex += 1
    return mex

def main():
    """Read input from stdin and solve the problem"""
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    for _ in range(Q):
        i, x = map(int, sys.stdin.readline().split())
        A[i-1] = x  # Update A
        print(mex(A))  # Print the mex of A

if __name__ == "__main__":
    main()