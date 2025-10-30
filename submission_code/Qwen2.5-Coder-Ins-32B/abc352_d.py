# YOUR CODE HERE
import sys
from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    P = list(map(int, input[2:]))

    # Create a list of indices for each value in P
    index_list = [[] for _ in range(N + 1)]
    for i, p in enumerate(P):
        index_list[p].append(i)

    # Initialize the minimum length to a large number
    min_length = N

    # Use a sliding window approach to find the minimum length of a good index sequence
    for indices in index_list:
        if len(indices) < K:
            continue
        window = deque(indices[:K])
        min_length = min(min_length, window[-1] - window[0])
        for i in range(K, len(indices)):
            window.popleft()
            window.append(indices[i])
            min_length = min(min_length, window[-1] - window[0])

    print(min_length)

if __name__ == "__main__":
    main()