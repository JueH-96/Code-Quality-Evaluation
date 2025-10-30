# YOUR CODE HERE
MOD = 998244353

def apply_operation(sequence):
    new_sequence = [0] * len(sequence)
    current_sum = 0
    for i in range(len(sequence)):
        current_sum += sequence[i]
        new_sequence[i] = current_sum % 2
    return new_sequence

def calculate_f(seq1, seq2):
    if seq1 == seq2:
        return 0
    
    operations = 0
    while operations < len(seq1):
        operations += 1
        seq1 = apply_operation(seq1)
        if seq1 == seq2:
            return operations
    
    return 0

def main():
    N, M = map(int, input().split())
    sequences = [list(map(int, input().split())) for _ in range(N)]
    
    total_sum = 0
    for i in range(N):
        for j in range(i, N):
            total_sum += calculate_f(sequences[i], sequences[j])
    
    print(total_sum % MOD)

if __name__ == "__main__":
    main()