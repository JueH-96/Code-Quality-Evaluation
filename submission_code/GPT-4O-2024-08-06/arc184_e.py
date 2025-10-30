# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    M = int(data[1])
    
    sequences = []
    index = 2
    for _ in range(N):
        sequences.append(list(map(int, data[index:index+M])))
        index += M
    
    def transform(sequence):
        transformed = [0] * M
        cumulative_sum = 0
        for i in range(M):
            cumulative_sum += sequence[i]
            transformed[i] = cumulative_sum % 2
        return transformed
    
    transformed_sequences = [transform(seq) for seq in sequences]
    
    def f(i, j):
        if transformed_sequences[i] == transformed_sequences[j]:
            return 0
        # Simulate the operation to see if they can become identical
        original_i = sequences[i][:]
        original_j = sequences[j][:]
        
        for x in range(1, M + 1):
            # Apply the operation once
            for k in range(M):
                original_i[k] = sum(original_i[:k+1]) % 2
                original_j[k] = sum(original_j[:k+1]) % 2
            
            if original_i == original_j:
                return x
        
        return 0
    
    result = 0
    for i in range(N):
        for j in range(i, N):
            result += f(i, j)
            result %= MOD
    
    print(result)

if __name__ == "__main__":
    main()