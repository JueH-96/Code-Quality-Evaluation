MOD = 998244353

def read_input():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    sequences = [tuple(map(int, data[i * M + 2:(i + 1) * M + 2])) for i in range(N)]

    return N, M, sequences

def transform_sequence(seq):
    transformed = []
    current_sum = 0
    for bit in seq:
        current_sum = (current_sum + bit) % 2
        transformed.append(current_sum)
    return tuple(transformed)

def find_min_operations(seq1, seq2):
    if seq1 == seq2:
        return 0
    for x in range(1, len(seq1) + 1):
        if seq1 == seq2:
            return x
        seq2 = transform_sequence(seq2)
    return 0

def main():
    N, M, sequences = read_input()
    transformed_sequences = [transform_sequence(seq) for seq in sequences]

    total_sum = 0
    for i in range(N):
        for j in range(i, N):
            total_sum = (total_sum + find_min_operations(transformed_sequences[i], transformed_sequences[j])) % MOD

    print(total_sum)

if __name__ == "__main__":
    main()