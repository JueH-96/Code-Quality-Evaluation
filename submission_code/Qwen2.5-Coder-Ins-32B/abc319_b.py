# YOUR CODE HERE
import sys

def solve(N):
    result = []
    for i in range(N + 1):
        smallest_j = None
        for j in range(1, 10):
            if N % j == 0 and i % (N // j) == 0:
                if smallest_j is None or j < smallest_j:
                    smallest_j = j
        if smallest_j is not None:
            result.append(str(smallest_j))
        else:
            result.append('-')
    return ''.join(result)

N = int(sys.stdin.read().strip())
print(solve(N))