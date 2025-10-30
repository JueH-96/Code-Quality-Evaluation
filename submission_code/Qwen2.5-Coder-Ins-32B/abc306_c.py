# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    # Dictionary to store the indices of each number
    indices = {}
    for i, num in enumerate(A):
        if num not in indices:
            indices[num] = []
        indices[num].append(i + 1)  # Convert to 1-based index

    # Calculate f(i) for each i
    f = {i: indices[i][1] for i in range(1, N + 1)}

    # Sort 1, 2, ..., N based on f(i)
    sorted_indices = sorted(range(1, N + 1), key=lambda x: f[x])

    # Print the result
    print(" ".join(map(str, sorted_indices)))

if __name__ == "__main__":
    main()