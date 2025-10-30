# Read the input
N, K = map(int, input().split())
S = input()

# Initialize the maximum number of strawberries to 0
max_strawberries = 0

# Iterate through the teeth
i = 0
while i < N:
    # Find the next sequence of K consecutive healthy teeth
    j = i
    while j < N and S[j] == 'O':
        j += 1
    
    # If we found a sequence of K consecutive healthy teeth, eat a strawberry
    if j - i >= K:
        max_strawberries += 1
        
        # Mark the K teeth as unhealthy
        for k in range(i, i + K):
            S = S[:k] + 'X' + S[k+1:]
        
        # Move to the next position
        i = j
    else:
        # Move to the next position
        i += 1

# Print the maximum number of strawberries
print(max_strawberries)