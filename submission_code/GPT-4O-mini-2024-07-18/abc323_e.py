def mod_inverse(a, p):
    return pow(a, p - 2, p)

def probability_song_playing(N, X, T):
    MOD = 998244353
    total_time = sum(T)
    target_time = X + 0.5

    # Calculate the probability that song 1 is playing at time target_time
    prob_numerator = 0
    prob_denominator = 1

    # We need to consider all possible times when song 1 could be playing
    for k in range(0, N):
        time_played = k * total_time + T[0]
        if time_played > target_time:
            break
        if time_played <= target_time:
            prob_numerator += 1

    # The total number of possible states is the sum of all songs played
    prob_denominator = 2 * total_time

    # The probability is prob_numerator / prob_denominator
    # We need to find z such that (prob_denominator * z) % MOD = prob_numerator
    prob_numerator %= MOD
    prob_denominator %= MOD

    # Calculate the modular inverse of prob_denominator
    prob_denominator_inv = mod_inverse(prob_denominator, MOD)

    # Calculate the result
    result = (prob_numerator * prob_denominator_inv) % MOD
    return result

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
T = list(map(int, data[2:N+2]))

result = probability_song_playing(N, X, T)
print(result)