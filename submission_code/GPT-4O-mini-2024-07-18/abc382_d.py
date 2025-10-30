def generate_sequences(N, M):
    results = []

    def backtrack(current_sequence, start):
        if len(current_sequence) == N:
            if current_sequence[-1] <= M:
                results.append(current_sequence[:])
            return
        
        for next_value in range(start, M + 1):
            if len(current_sequence) == 0 or (current_sequence[-1] + 10 <= next_value):
                current_sequence.append(next_value)
                backtrack(current_sequence, next_value)
                current_sequence.pop()

    backtrack([], 1)
    return results

def main():
    import sys
    input = sys.stdin.read
    N, M = map(int, input().strip().split())
    
    sequences = generate_sequences(N, M)
    
    print(len(sequences))
    for seq in sequences:
        print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()