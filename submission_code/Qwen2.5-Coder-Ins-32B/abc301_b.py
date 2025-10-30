# YOUR CODE HERE
import sys

def solve():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    while True:
        modified = False
        i = 0
        while i < len(A) - 1:
            if abs(A[i] - A[i + 1]) != 1:
                if A[i] < A[i + 1]:
                    A[i:i + 1] = list(range(A[i], A[i + 1] + 1))
                else:
                    A[i:i + 1] = list(range(A[i], A[i + 1] - 1, -1))
                modified = True
                break
            i += 1
        if not modified:
            break

    print(" ".join(map(str, A)))

solve()