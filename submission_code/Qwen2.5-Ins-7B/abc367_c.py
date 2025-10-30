# YOUR CODE HERE
from itertools import product

def find_sequences(N, K, R):
    sequences = []
    for seq in product(*[range(1, R[i]+1) for i in range(N)]):
        if sum(seq) % K == 0:
            sequences.append(seq)
    return sequences

def main():
    N, K = map(int, input().split())
    R = list(map(int, input().split()))
    sequences = find_sequences(N, K, R)
    print(len(sequences))
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()