def max_strawberries(N, K, S):
    count = 0
    i = 0
    
    while i <= N - K:
        # Check if there are K consecutive 'O's
        if S[i:i + K] == 'O' * K:
            count += 1
            # After eating a strawberry, mark these K teeth as 'X'
            S = S[:i] + 'X' * K + S[i + K:]
            # Move i back to check for overlapping K 'O's
            i = max(i - K + 1, 0)
        else:
            i += 1
            
    return count

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
N, K = map(int, data[0].split())
S = data[1]

# Calculate the result
result = max_strawberries(N, K, S)

# Print the result
print(result)