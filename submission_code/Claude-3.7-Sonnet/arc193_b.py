def count_indegree_sequences(N, s):
    MOD = 998244353
    
    # Count the number of edges
    E = N  # Number of edges in the cycle
    for i in range(N):
        if s[i] == '1':
            E += 1  # Add edges connecting to vertex N
    
    # Calculate the degree of each vertex
    degrees = [2] * N  # Each vertex in the cycle is connected to 2 other vertices
    for i in range(N):
        if s[i] == '1':
            degrees[i] += 1  # Add an edge to vertex N
    degrees.append(sum(1 for i in range(N) if s[i] == '1'))  # Degree of vertex N
    
    # Use dynamic programming to calculate the coefficient of x^E
    dp = [0] * (E + 1)
    dp[0] = 1
    
    for d in degrees:
        new_dp = [0] * (E + 1)
        for i in range(E + 1):
            for j in range(min(d + 1, E - i + 1)):
                new_dp[i + j] = (new_dp[i + j] + dp[i]) % MOD
        dp = new_dp
    
    return dp[E]

def main():
    N = int(input().strip())
    s = input().strip()
    print(count_indegree_sequences(N, s))

if __name__ == "__main__":
    main()