def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    MOD = 998244353
    results = []
    
    # To store the total score for each m
    total_scores = [0] * (N + 1)
    
    # To store the contribution of each number
    contribution = [0] * (100001)
    
    for m in range(1, N + 1):
        # Update contribution for the current A[m-1]
        current = A[m - 1]
        
        # Calculate the contribution of the current number
        for j in range(1, 100001):
            if contribution[j] > 0:
                total_scores[m] = (total_scores[m] + contribution[j] * gcd(current, j)) % MOD
        
        # Update the contribution for the current number
        contribution[current] = (contribution[current] + 1) % MOD
        
        # Add the current total score to results
        results.append(total_scores[m])
    
    # Print the results
    print('
'.join(map(str, results)))