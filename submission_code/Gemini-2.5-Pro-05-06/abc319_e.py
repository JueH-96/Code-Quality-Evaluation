import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    
    # P_vals[k], T_vals[k] store parameters for bus from stop k+1 to stop k+2.
    # These correspond to P_{k+1}, T_{k+1} in 1-indexed problem statement.
    P_vals = [0] * (N - 1)
    T_vals = [0] * (N - 1)
    for i in range(N - 1):
        P_vals[i], T_vals[i] = map(int, sys.stdin.readline().split())

    Q_count = int(sys.stdin.readline())
    
    LCM = 840 # LCM of integers from 1 to 8
    
    # F_table[t_rem] will store the arrival time at bus stop N,
    # assuming arrival at bus stop 1 was at time t_rem (where t_rem in [0, LCM-1]).
    # Initialize F_table[t_rem] = t_rem, representing arrival at bus stop 1.
    F_table = list(range(LCM))

    # Simulate the bus journey for initial arrival times t_rem at stop 1.
    # After k-th iteration of this outer loop (0-indexed P_vals/T_vals),
    # F_table[t_rem] will store arrival time at bus stop k+2.
    for k in range(N - 1): # k from 0 to N-2
        P_k = P_vals[k]
        T_k = T_vals[k]
        for t_rem_idx in range(LCM):
            # current_time is arrival time at bus stop k+1
            current_time = F_table[t_rem_idx] 
            
            wait_time = (-current_time) % P_k
            
            departure_time = current_time + wait_time
            # arrival_at_next_stop is arrival time at bus stop k+2
            arrival_at_next_stop = departure_time + T_k
            F_table[t_rem_idx] = arrival_at_next_stop
            
    # Now, F_table[t_rem] stores arrival time at bus stop N,
    # given that arrival at bus stop 1 was t_rem.

    results_buffer = []
    for _ in range(Q_count):
        q_val = int(sys.stdin.readline())
        
        # Time of arrival at bus stop 1
        arrival_at_stop1 = q_val + X
        
        num_lcm_blocks = arrival_at_stop1 // LCM
        t_rem_arrival_stop1 = arrival_at_stop1 % LCM
        
        # F(t) = arrival_at_N given arrival_at_stop1 = t.
        # We have F(t_rem + K*LCM) = F(t_rem) + K*LCM.
        # arrival_at_stopN = F(arrival_at_stop1)
        #                  = F(t_rem_arrival_stop1 + num_lcm_blocks * LCM)
        #                  = F(t_rem_arrival_stop1) + num_lcm_blocks * LCM
        # F(t_rem_arrival_stop1) is stored in F_table[t_rem_arrival_stop1].
        
        base_arrival_at_stopN = F_table[t_rem_arrival_stop1]
        total_arrival_at_stopN = base_arrival_at_stopN + num_lcm_blocks * LCM
        
        # Time of arrival at Aoki's house
        final_arrival_time = total_arrival_at_stopN + Y
        results_buffer.append(str(final_arrival_time))
        
    sys.stdout.write("
".join(results_buffer) + "
")

solve()