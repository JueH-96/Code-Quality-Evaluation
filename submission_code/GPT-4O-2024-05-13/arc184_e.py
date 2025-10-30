# YOUR CODE HERE
import sys
input = sys.stdin.read

def transform(sequence):
    transformed = [sequence[0]]
    for i in range(1, len(sequence)):
        transformed.append((transformed[-1] + sequence[i]) % 2)
    return transformed

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    sequences = []
    index = 2
    for _ in range(N):
        sequences.append(list(map(int, data[index:index + M])))
        index += M
    
    transformed_sequences = [transform(seq) for seq in sequences]
    
    def f(i, j):
        if sequences[i] == sequences[j]:
            return 0
        for x in range(1, M + 1):
            if transformed_sequences[i] == transformed_sequences[j]:
                return x
            transformed_sequences[i] = transform(transformed_sequences[i])
            transformed_sequences[j] = transform(transformed_sequences[j])
        return 0
    
    result = 0
    for i in range(N):
        for j in range(i, N):
            result += f(i, j)
            result %= 998244353
    
    print(result)

if __name__ == "__main__":
    main()