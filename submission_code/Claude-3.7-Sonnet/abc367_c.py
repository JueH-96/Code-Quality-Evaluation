def generate_sequences(n, k, max_values):
    result = []
    
    def backtrack(index, current, current_sum):
        if index == n:
            if current_sum % k == 0:
                result.append(current[:])
            return
        
        for i in range(1, max_values[index] + 1):
            current.append(i)
            backtrack(index + 1, current, current_sum + i)
            current.pop()
    
    backtrack(0, [], 0)
    return result

# Read input
n, k = map(int, input().split())
max_values = list(map(int, input().split()))

# Generate and print sequences
sequences = generate_sequences(n, k, max_values)
for seq in sequences:
    print(' '.join(map(str, seq)))