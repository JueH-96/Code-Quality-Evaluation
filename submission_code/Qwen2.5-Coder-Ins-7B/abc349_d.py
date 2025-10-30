# YOUR CODE HERE
import sys

def find_good_sequences(L, R):
    sequences = []
    while L < R:
        i = 0
        while (1 << i) * (L // (1 << i)) < L:
            i += 1
        l = (1 << i) * (L // (1 << i))
        r = (1 << i) * ((L // (1 << i)) + 1)
        sequences.append((l, r))
        L = r
    return sequences

def main():
    input = sys.stdin.read
    data = input().split()
    L = int(data[0])
    R = int(data[1])
    
    sequences = find_good_sequences(L, R)
    print(len(sequences))
    for seq in sequences:
        print(seq[0], seq[1])

if __name__ == "__main__":
    main()