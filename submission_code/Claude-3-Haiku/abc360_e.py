import math

def expected_black_ball_position(N, K):
    # Initialize the expected position of the black ball
    expected_pos = 1
    
    # Perform the K operations
    for _ in range(K):
        # Choose two random positions
        a = random.randint(1, N)
        b = random.randint(1, N)
        
        # Swap the balls if the positions are different
        if a != b:
            # Update the expected position of the black ball
            expected_pos = (expected_pos - 1 + (b - a) * (a <= expected_pos < b)) % (N - 1) + 1
    
    # Calculate the result modulo 998244353
    result = expected_pos % 998244353
    
    # Find the unique integer R such that R * Q ≡ P (mod 998244353)
    Q = 1
    P = result
    R = pow(Q, 998244351, 998244353)
    return R