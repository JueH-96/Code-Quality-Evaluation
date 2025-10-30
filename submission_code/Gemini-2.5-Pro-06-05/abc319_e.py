# YOUR CODE HERE
import sys

def solve():
    """
    Solves the Takahashi's Journey problem.
    The solution uses precomputation based on the periodicity of the bus journey time.
    """
    
    # Use fast I/O
    input = sys.stdin.readline

    # Read problem parameters
    try:
        line = input().split()
        if not line: return # Handles empty input at EOF
        N, X, Y = map(int, line)
    except (ValueError, IndexError):
        return

    # Read bus schedules (P_i, T_i) for stops i=1 to N-1
    buses = []
    for _ in range(N - 1):
        line = input().split()
        P = int(line[0])
        T = int(line[1])
        buses.append((P, T))

    # --- Key Insight & Precomputation ---
    # The waiting time at a bus stop `i` depends on the arrival time `t` modulo `P_i`.
    # The periods `P_i` are small (1 to 8). The least common multiple (LCM) of
    # all possible periods {1, ..., 8} is 840.
    # Let L = 840.
    # The total time taken for the bus journey from stop 1 to stop N is periodic
    # with a period of L with respect to the arrival time at stop 1.
    #
    # Let `BusDuration(t)` be the time taken to travel from arrival at stop 1 at time `t`
    # to arrival at stop N. It can be shown that `BusDuration(t)` only depends on `t % L`.
    # We can precompute `BusDuration(s)` for s = 0, 1, ..., L-1.
    
    L = 840
    
    # Precomputation table: bus_duration[s] stores the duration of the bus journey
    # from stop 1 to N, given an arrival at stop 1 at time `s`.
    bus_duration = [0] * L
    
    for s_start in range(L):
        current_time = s_start
        # Simulate the journey from bus stop 1 to N for a starting time `s_start`
        for p, t in buses:
            # Calculate waiting time until the next multiple of p.
            # If current_time is already a multiple of p, wait is 0.
            # Otherwise, wait is p - (current_time % p).
            # This can be concisely written as (p - current_time % p) % p.
            wait = (p - current_time % p) % p
            
            # Update time to departure time
            current_time += wait
            
            # Add bus travel time
            current_time += t
            
        # The duration of the journey from bus stop 1 to N is (final_time - start_time)
        bus_duration[s_start] = current_time - s_start
        
    # --- Process Queries ---
    Q = int(input())
    answers = []
    for _ in range(Q):
        q_i = int(input())
        
        # 1. Leave house at q_i, walk to bus stop 1.
        # Arrival time at bus stop 1:
        time_at_stop1 = q_i + X
        
        # 2. Travel from bus stop 1 to N.
        # The duration depends on `time_at_stop1 % L`.
        s = time_at_stop1 % L
        duration_on_buses = bus_duration[s]
        
        # Arrival time at bus stop N:
        time_at_stopN = time_at_stop1 + duration_on_buses
        
        # 3. Walk from bus stop N to Aoki's house.
        final_arrival_time = time_at_stopN + Y
        
        # Store answer to print all at once later (can be slightly faster)
        answers.append(str(final_arrival_time))
    
    # Print all answers, separated by newlines.
    sys.stdout.write("
".join(answers) + "
")

solve()