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
    lcm = 840
    # Initialize a list to store the earliest arrival time at each bus stop for each starting time modulo lcm
    # arrival_time[k][i] represents the earliest time to reach bus stop i when starting at time k modulo lcm
    arrival_time = [[0] * N for _ in range(lcm)]
    
    for k in range(lcm):
        current_time = k
        arrival_time[k][0] = current_time + X
        for i in range(N-1):
            current_time = arrival_time[k][i]
            # Find the next bus departure time
            # The bus departs at the smallest multiple of P_i >= current_time
            if current_time % P[i] == 0:
                departure_time = current_time
            else:
                departure_time = current_time + (P[i] - current_time % P[i])
            arrival_time[k][i+1] = departure_time + T[i]
    
    for q in queries:
        k = q % lcm
        total_time = arrival_time[k][N-1] + Y
        print(total_time)

if __name__ == "__main__":
    main()