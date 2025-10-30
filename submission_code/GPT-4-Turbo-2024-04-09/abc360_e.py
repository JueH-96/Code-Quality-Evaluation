def modinv(a, m):
    """ Compute modular inverse of a under modulo m """
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += m0
    return x1

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    
    MOD = 998244353
    
    # The expected position of the black ball after K operations
    if N == 1:
        # If there's only one ball, it never moves
        print(1)
        return
    
    # The probability that any two distinct positions are chosen to swap
    p_swap = 2 * K / (N * (N - 1))
    
    # The probability that a specific position remains unchanged after one operation
    p_stay = 1 - p_swap
    
    # The expected position of the black ball after K operations
    # E[x] = 1 * p1 + 2 * p2 + ... + N * pN
    # where pi is the probability that the black ball is at position i after K operations
    
    # Initially, the black ball is at position 1
    # Probability distribution of the black ball's position
    prob = [0] * N
    prob[0] = 1
    
    # Update the probability distribution for each operation
    for _ in range(K):
        new_prob = [0] * N
        for i in range(N):
            if prob[i] > 0:
                # The probability that it stays at the current position
                new_prob[i] += prob[i] * p_stay
                # The probability that it moves to any other position
                spread_prob = prob[i] * p_swap / (N - 1)
                for j in range(N):
                    if i != j:
                        new_prob[j] += spread_prob
        prob = new_prob
    
    # Calculate the expected position
    expected_position = sum((i + 1) * prob[i] for i in range(N))
    
    # Since the expected value is a rational number, we need to find its representation modulo 998244353
    # The expected value is a sum of terms (i+1) * prob[i], where prob[i] is a rational number
    # We need to find the modular inverse of the denominator to convert it into an integer modulo 998244353
    
    # Calculate the denominator of the expected position (which is a sum of probabilities)
    denominator = sum(prob)
    numerator = sum((i + 1) * prob[i] for i in range(N))
    
    # Convert to integers under modulo
    numerator = int(numerator * (N * (N - 1))**K) % MOD
    denominator = int(denominator * (N * (N - 1))**K) % MOD
    
    # Find the modular inverse of the denominator
    denominator_inv = modinv(denominator, MOD)
    
    # Calculate the result
    result = (numerator * denominator_inv) % MOD
    
    print(result)