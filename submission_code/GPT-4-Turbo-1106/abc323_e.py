MOD = 998244353

def modinv(a, m):
    m0, y, x = m, 0, 1
    while a > 1:
        q = a // m
        t = m
        m = a % m
        a = t
        t = y
        y = x - q * y
        x = t
    if x < 0:
        x += m0
    return x

def solve(N, X, T):
    # Calculate the sum of all song lengths
    total_length = sum(T)
    
    # Initialize the probability of song 1 being played at time X + 0.5
    probability = 0
    
    # Calculate the probability
    for i in range(N):
        if T[i] > X:
            continue
        # Number of times the song can be played in X seconds
        count = (X - T[i]) // total_length
        # Add the probability of song 1 being played after count full cycles and song i
        probability += count + 1
        # Subtract the probability of song 1 being played after (count + 1) full cycles and song i
        if T[i] + (count + 1) * total_length <= X:
            probability -= 1
    
    # Multiply by the modular inverse of N to get the final probability
    probability = (probability * modinv(N, MOD)) % MOD
    
    # Multiply by the modular inverse of N again because each song has a 1/N chance of being played
    probability = (probability * modinv(N, MOD)) % MOD
    
    return probability

# Read input
N, X = map(int, input().split())
T = list(map(int, input().split()))

# Solve the problem and print the answer
print(solve(N, X, T))