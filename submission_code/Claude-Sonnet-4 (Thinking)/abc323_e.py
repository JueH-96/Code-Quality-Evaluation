from fractions import Fraction

def solve():
    n, x = map(int, input().split())
    t = list(map(int, input().split()))
    
    # dp[time] = {song: probability that song starts playing at time}
    dp = {0: {i: Fraction(1, n) for i in range(n)}}
    
    # Simulation up to a reasonable bound
    max_time = x + max(t) + 1
    for time in range(max_time):
        if time not in dp:
            continue
        
        new_events = {}
        for song, prob in dp[time].items():
            if prob == 0:
                continue
            end_time = time + t[song]
            if end_time < max_time:
                if end_time not in new_events:
                    new_events[end_time] = {i: Fraction(0) for i in range(n)}
                for next_song in range(n):
                    new_events[end_time][next_song] += prob / n
        
        for end_time, song_probs in new_events.items():
            if end_time not in dp:
                dp[end_time] = {i: Fraction(0) for i in range(n)}
            for song, prob in song_probs.items():
                dp[end_time][song] += prob
    
    # Check what's playing at time x + 0.5
    total_prob = Fraction(0)
    for start_time in dp:
        if start_time <= x:
            for song, prob in dp[start_time].items():
                if prob > 0 and start_time + t[song] > x:
                    if song == 0:  # song 1 (0-indexed)
                        total_prob += prob
    
    return total_prob

def mod_inverse(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, y = extended_gcd(a, m)
    return (x % m + m) % m

def main():
    prob = solve()
    numerator = prob.numerator
    denominator = prob.denominator
    
    MOD = 998244353
    inv_denominator = mod_inverse(denominator, MOD)
    result = (numerator * inv_denominator) % MOD
    print(result)

if __name__ == "__main__":
    main()