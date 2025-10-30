import sys
from itertools import product

def generate_sequences(N, M):
    sequences = []
    for seq in product(range(1, M + 1), repeat=N):
        if all(seq[i] >= seq[i-1] + 10 for i in range(1, N)) and seq[-1] <= M:
            sequences.append(seq)
    sequences.sort()
    return sequences

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])

    sequences = generate_sequences(N, M)
    print(len(sequences))
    for seq in sequences:
        print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()