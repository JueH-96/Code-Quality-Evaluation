# YOUR CODE HERE
def generate_sequences(N, M):
    def backtrack(sequence, last_value):
        if len(sequence) == N:
            sequences.append(sequence[:])
            return
        
        start = last_value + 10 if sequence else 1
        for next_value in range(start, M + 1):
            sequence.append(next_value)
            backtrack(sequence, next_value)
            sequence.pop()
    
    sequences = []
    backtrack([], 0)
    return sequences

def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, M = map(int, data.split())
    
    sequences = generate_sequences(N, M)
    print(len(sequences))
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()