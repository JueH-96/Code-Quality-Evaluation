# YOUR CODE HERE
import sys
import threading

MOD = 998244353

def main():
    import sys
    import bisect
    import math
    import heapq
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    P = []
    V = []
    for _ in range(Q):
        pi, vi = map(int, sys.stdin.readline().split())
        P.append(pi)
        V.append(vi)

    N = N
    Q = Q
    P = P
    V = V

    # For each position, collect the list of operations that can affect it
    position_events = [[] for _ in range(N+1)]  # positions are from 1 to N

    for i in range(Q):
        pi = P[i]
        vi = V[i]
        # Choices:
        # Choice 0 (left): positions 1 to P_i
        # Choice 1 (right): positions P_i to N
        # Under choice 0, positions 1 to P_i are affected
        # Under choice 1, positions P_i to N are affected
        # For each position, we record (time i, choice c_i = 0 or 1, V_i)

        for c in [0,1]:
            if c == 0:
                positions = range(1, pi+1)
            else:
                positions = range(pi, N+1)
            for pos in positions:
                position_events[pos].append( (i, c, vi) )
    result = 1
    for pos in range(1,N+1):
        events = position_events[pos]
        if not events:
            continue
        # For this position, we need to process the events and count the number of ways
        # to select choices c_i at times i, such that the sequence of V_i assigned to position pos is non-decreasing
        # However, at each time i, only one choice c_i is selected
        # So we need to model the possible choices per time i
        # Since total number of events per position can be up to 2Q
        # We can process DP over times i

        # First, collect events per time i
        events_per_time = {}
        for event in events:
            i, c, vi = event
            if i not in events_per_time:
                events_per_time[i] = []
            events_per_time[i].append( (c, vi) )
        # Since only one choice per time i, and per position pos, we need to consider the combinations
        times = sorted(events_per_time.keys())
        # Collect all V_i that can be assigned to position pos, compress them
        V_values = set()
        for event_list in events_per_time.values():
            for c, vi in event_list:
                V_values.add(vi)
        V_values = sorted(V_values)
        V_index = {v:i for i,v in enumerate(V_values)}
        K = len(V_values)
        dp = [{}]  # dp[i][v_index] = number of ways
        dp[0][ -1 ] = 1  # Start with initial value 0, which is index -1
        for idx, time in enumerate(times):
            dp_next = {}
            event_list = events_per_time[time]
            choices = [0,1]
            # For each previous value v_prev_index
            for v_prev_index in dp[idx]:
                count = dp[idx][v_prev_index]
                v_prev = V_values[v_prev_index] if v_prev_index != -1 else 0
                # For the current time, we can select one choice c_i among 0 or 1
                # For each c_i, if it updates position pos with vi >= v_prev, we can proceed
                # We need to consider only one c_i per time
                total_ways = 0
                for c in [0,1]:
                    # Check if choice c is possible for this position at this time
                    vi = None
                    for (c_event, vi_event) in event_list:
                        if c_event == c:
                            vi = vi_event
                            break
                    if vi is not None:
                        # This choice c_i updates position pos with vi
                        if (v_prev_index == -1) or (vi >= v_prev):
                            # We can proceed
                            v_index = V_index[vi]
                            if v_index not in dp_next:
                                dp_next[v_index] = 0
                            dp_next[v_index] = (dp_next[v_index] + count) % MOD
                    else:
                        # This choice c does not update position pos
                        if v_prev_index not in dp_next:
                            dp_next[v_prev_index] = 0
                        dp_next[v_prev_index] = (dp_next[v_prev_index] + count) % MOD
                # But we only select one c_i per time i
            dp.append(dp_next)
        total = sum(dp[-1].values()) % MOD
        result = (result * total) % MOD
    print(result)