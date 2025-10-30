def mod_inverse(a, m):
    # Extended Euclidean Algorithm to find modular inverse
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

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    T = list(map(int, data[2:2+N]))
    
    MOD = 998244353
    
    total_duration = sum(T)
    
    # Calculate the time point we are interested in
    target_time = X + 0.5
    
    # Calculate the number of full cycles and the remaining time
    full_cycles = int(target_time // total_duration)
    remaining_time = target_time - full_cycles * total_duration
    
    # Check if song 1 is playing during the remaining time
    current_time = 0
    is_song_1_playing = False
    for i in range(N):
        current_time += T[i]
        if remaining_time < current_time:
            if i == 0:
                is_song_1_playing = True
            break
    
    # Calculate the probability
    if is_song_1_playing:
        y = 1
    else:
        y = 0
    x = N
    
    # Find the modular inverse of x modulo MOD
    x_inv = mod_inverse(x, MOD)
    
    # Calculate the result
    result = (y * x_inv) % MOD
    
    print(result)