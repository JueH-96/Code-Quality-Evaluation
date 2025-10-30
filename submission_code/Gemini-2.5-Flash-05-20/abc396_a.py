import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # The problem asks to check if A_i = A_{i+1} = A_{i+2}
    # This means we need to compare elements at index i, i+1, and i+2.
    # The maximum index we can access is N-1.
    # So, for A[i+2] to be valid, i+2 must be less than or equal to N-1.
    # This implies i <= N-1-2, which means i <= N-3.
    # Therefore, the loop for 'i' should go from 0 up to N-3 (inclusive).
    # In Python's range(end), 'end' is exclusive, so range(N-2) will cover
    # indices from 0 to N-3.

    found_triple = False
    for i in range(N - 2):
        if A[i] == A[i+1] and A[i+1] == A[i+2]:
            found_triple = True
            break # Found a triplet, no need to check further

    if found_triple:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()