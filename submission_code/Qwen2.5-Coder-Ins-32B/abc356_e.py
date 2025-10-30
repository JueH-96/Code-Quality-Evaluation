# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    # Sort the array and keep track of the original indices
    sorted_A = sorted((a, i) for i, a in enumerate(A))
    counts = [0] * (10**6 + 1)
    total = 0

    for a, _ in sorted_A:
        # For each element, count how many times it has been seen so far
        for j in range(a, 10**6 + 1, a):
            total += counts[j]
        counts[a] += 1

    print(total)

if __name__ == "__main__":
    main()