import itertools

def find_sequences(N, K, R):
    sequences = []
    for seq in itertools.product(*[range(1, r + 1) for r in R]):
        if sum(seq) % K == 0:
            sequences.append(seq)
    sequences.sort()
    return sequences

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:]))
    
    sequences = find_sequences(N, K, R)
    
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()