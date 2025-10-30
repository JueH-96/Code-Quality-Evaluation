MOD = 998244353

def modinv(a, mod=MOD):
    return pow(a, mod - 2, mod)

def solve(N, K):
    # The expected value of x is the sum of probabilities of the black ball
    # being in each position times the position index.
    # The probability of the black ball being in the first position after K
    # operations is (1 - 1/N) ** K.
    # For other positions, we need to consider the probability of the black ball
    # moving to that position for the first time and then staying there.
    
    # Calculate the probability of the black ball staying in the first position
    prob_stay_first = pow(N - 1, K, MOD) * modinv(pow(N, K, MOD), MOD) % MOD
    
    # Initialize result with the probability of the black ball staying in the first position
    result = prob_stay_first
    
    # Initialize the probability of the black ball moving to a new position for the first time
    prob_move = 1
    for i in range(1, N):
        # Calculate the probability of the black ball moving to the i-th position for the first time
        prob_move = prob_move * (N - i) % MOD * modinv(N, MOD) % MOD
        # Update the result with the probability of the black ball being in the i-th position
        result = (result + prob_move * pow(N - 1, K - 1, MOD) * modinv(pow(N, K - 1, MOD), MOD) % MOD * i) % MOD
    
    return result

# Read input
N, K = map(int, input().split())

# Solve the problem and print the result
print(solve(N, K))