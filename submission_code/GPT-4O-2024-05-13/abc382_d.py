# YOUR CODE HERE
def generate_sequences(N, M):
    def backtrack(start, sequence):
        if len(sequence) == N:
            results.append(sequence[:])
            return
        for i in range(start, M + 1):
            if not sequence or sequence[-1] + 10 <= i:
                sequence.append(i)
                backtrack(i + 1, sequence)
                sequence.pop()
    
    results = []
    backtrack(1, [])
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    sequences = generate_sequences(N, M)
    print(len(sequences))
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()