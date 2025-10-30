# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    # Dictionary to store the indices of each value
    indices = defaultdict(list)
    for i in range(N):
        indices[A[i]].append(i)

    count = 0
    for values in indices.values():
        m = len(values)
        if m < 2:
            continue
        # Calculate the number of valid (i, j, k) triples
        for j in range(1, m):
            i_count = values[j] - values[j-1] - 1
            k_count = N - values[j] - 1
            count += i_count * k_count

    print(count)

if __name__ == "__main__":
    main()