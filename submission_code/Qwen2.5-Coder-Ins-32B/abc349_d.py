# YOUR CODE HERE
import sys

def find_good_sequences(L, R):
    sequences = []
    while L < R:
        # Find the largest power of 2 such that 2^i * j <= L < 2^i * (j+1) < R
        i = 0
        while (L + 1) & (1 << i) == 0:
            i += 1
        # Calculate the next boundary
        next_boundary = (L >> i + 1) + 1
        next_boundary <<= i + 1
        if next_boundary > R:
            next_boundary = R
        sequences.append((L, next_boundary))
        L = next_boundary
    return sequences

def main():
    import sys
    input = sys.stdin.read().strip().split()
    L = int(input[0])
    R = int(input[1])
    sequences = find_good_sequences(L, R)
    print(len(sequences))
    for l, r in sequences:
        print(l, r)

if __name__ == "__main__":
    main()