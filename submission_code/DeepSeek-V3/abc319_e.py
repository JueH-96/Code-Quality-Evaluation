import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    X = int(data[idx])
    idx += 1
    Y = int(data[idx])
    idx += 1
    P = []
    T = []
    for _ in range(N-1):
        P.append(int(data[idx]))
        idx += 1
        T.append(int(data[idx]))
        idx += 1
    Q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        queries.append(int(data[idx]))
        idx += 1
    
    # Precompute the earliest arrival time at each bus stop for each possible starting time modulo LCM of P_i
    # Since P_i <= 8, the LCM of all P_i is at most 840 (LCM of 1..8)
    lcm = 840  # LCM of 1..8 is 840
    
    # Initialize a list to store the earliest arrival time at each bus stop for each possible starting time modulo lcm
    # dp[i][j] represents the earliest arrival time at bus stop i when starting at time j modulo lcm
    dp = [[float('inf')] * lcm for _ in range(N)]
    
    # Initialize the first bus stop
    for t_mod in range(lcm):
        # Time to reach bus stop 1 is X
        # The earliest time to reach bus stop 1 is t_mod + X
        # But since t_mod is the starting time modulo lcm, we need to find the earliest time >= t_mod + X that is congruent to t_mod + X modulo lcm
        # Wait, no. The starting time is t_mod, and we need to find the earliest time to reach bus stop 1
        # The time to reach bus stop 1 is t_mod + X
        # So the earliest arrival time at bus stop 1 is t_mod + X
        dp[0][t_mod] = t_mod + X
    
    # Now, for each bus stop, compute the earliest arrival time at the next bus stop
    for i in range(N-1):
        p = P[i]
        t = T[i]
        for t_mod in range(lcm):
            current_time = dp[i][t_mod]
            # Find the earliest bus departure time >= current_time that is a multiple of p
            # The next bus departs at the smallest multiple of p >= current_time
            # So the departure time is ceil(current_time / p) * p
            departure_time = ((current_time + p - 1) // p) * p
            arrival_time = departure_time + t
            # Update the next bus stop's earliest arrival time
            next_t_mod = arrival_time % lcm
            if arrival_time < dp[i+1][next_t_mod]:
                dp[i+1][next_t_mod] = arrival_time
    
    # Now, for each query, compute the earliest arrival time at Aoki's house
    for q in queries:
        t_mod = q % lcm
        # The earliest arrival time at bus stop N is dp[N-1][t_mod]
        arrival_at_N = dp[N-1][t_mod]
        # The total time is arrival_at_N + Y
        total_time = arrival_at_N + Y
        print(total_time)

if __name__ == "__main__":
    main()