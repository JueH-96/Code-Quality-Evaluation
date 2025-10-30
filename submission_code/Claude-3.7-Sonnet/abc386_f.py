def min_operations(S, T, K):
    m, n = len(S), len(T)
    
    # If the difference in length is greater than K, it's impossible
    if abs(m - n) > K:
        return "No"
    
    # We'll use two arrays to remember just the previous and current rows
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i]  # Initialize the first column
        for j in range(1, n + 1):
            if S[i-1] == T[j-1]:
                curr.append(prev[j-1])
            else:
                curr.append(1 + min(prev[j],      # Delete
                                   curr[j-1],     # Insert
                                   prev[j-1]))    # Replace
        prev = curr
    
    # Check if the minimum operations required is less than or equal to K
    return "Yes" if prev[n] <= K else "No"

# Read inputs
K = int(input())
S = input()
T = input()

# Print output
print(min_operations(S, T, K))