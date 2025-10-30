import sys
from collections import Counter

# Use long long for costs to avoid overflow
INF = float('inf')

def dist(p1, p2, M):
    d = (p2 - p1) % M
    return min(d, M - d)

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Required shifts for each element A_i
    diffs = [(B[i] - A[i]) % M for i in range(N)]

    # The total cost for a global shift k is
    # C(k) = N * dist(0, k, M) + sum_{i=1..N} dist((B_i - A_i)%M, k, M).
    # Let d_i = (B_i - A_i)%M.
    # Cost = sum_{v in V} dist(v, k, M) where V = {0 (N times), d_1, ..., d_N}.

    # Collect all points in V
    all_points = []
    for _ in range(N):
        all_points.append(0)
    all_points.extend(diffs)

    # Count occurrences of each point in V
    point_counts = Counter(all_points)

    # Calculate C[0] = sum_{v in V} dist(v, 0, M)
    C_0 = sum(dist(v, 0, M) for v in all_points)
    min_cost = C_0

    # Calculate S[0] = C[1] - C[0] = sum_{v in V} (dist(v, 1, M) - dist(v, 0, M))
    S_0 = sum(dist(v, 1, M) - dist(v, 0, M) for v in all_points)

    # Calculate DeltaS[k] = S[k] - S[k-1] for k = 1 .. M-1
    # DeltaS[k] is non-zero at k = v and k = (v + M/2) % M for v in V.
    # DeltaS[v] = +2 * count(v)
    # DeltaS[(v + M/2) % M] = -2 * count(v)
    
    delta_S_values_arr = [0] * M # Stores DeltaS[k] for k = 0 .. M-1. DeltaS[0] is only relevant if we wrap around.

    for v, count in point_counts.items():
        # DeltaS at k=v is +2*count
        # This means S[v] = S[v-1] + 2*count
        # The change in S happens *at* k=v.
        # DeltaS[k] = S[k] - S[k-1]
        # DeltaS[v] = S[v] - S[v-1]
        # The position where S changes value from S(k-1) to S(k) is k.
        # S(k) slope changes at v and (v+M/2)%M.
        # S(k) = C(k+1)-C(k) changes at k where C'(k) changes.
        # C'(k) changes at v and (v+M/2)%M.
        # So S(k) changes at v and (v+M/2)%M.
        # DeltaS(k) = S(k) - S(k-1). DeltaS is non-zero at k where S changes.
        # S changes at k=v and k=(v+M/2)%M.
        # DeltaS[v] = +2 * count(v).
        # DeltaS[(v+M/2)%M] = -2 * count(v). This must be integer position.
        # If M is odd, (v+M/2)%M is X.5.
        # S(k) changes at X.5. S(X) vs S(X+1).
        # S(X+1)-S(X) should capture the -2*count change across X.5.
        # This change is split between DeltaS[X+1] and DeltaS[X].
        # DeltaS[X+1] = S[X+1]-S[X].
        # DeltaS[k] is sum_{v} count(v) * ( (d(v,k+1)-d(v,k)) - (d(v,k)-d(v,k-1)) ).
        # This second difference is +2 if k=v, -2 if k=(v+M/2)%M.

        # Change happens *at* position `v` for slope `S`.
        # This corresponds to DeltaS[v] = +2 * count.
        delta_S_values_arr[v] = delta_S_values_arr[v] + 2 * count

        # Change happens *at* position `(v+M/2)%M` for slope `S`.
        # This corresponds to DeltaS[(v+M/2)%M] = -2 * count.
        pos2_float = (v + M / 2.0)
        
        # Split the -2 * count change across the integer boundary if M is odd
        if M % 2 != 0:
            pos2_floor = int(pos2_float) % M
            pos2_ceil = (int(pos2_float) + 1) % M
            # Subtract count at floor and ceil positions
            delta_S_values_arr[pos2_floor] = delta_S_values_arr[pos2_floor] - count
            if pos2_floor != pos2_ceil: # Avoid double subtraction if floor == ceil (M=1)
                delta_S_values_arr[pos2_ceil] = delta_S_values_arr[pos2_ceil] - count
        else: # M is even, (v+M/2)%M is integer
            pos2_int = int(pos2_float) % M
            delta_S_values_arr[pos2_int] = delta_S_values_arr[pos2_int] - 2 * count


    # Sweep k from 0 to M-2 to compute C[k+1] and S[k+1]
    # current_k = 0, current_C = C_0, current_S = S_0
    # min_cost = C_0

    current_C = C_0 # C[0]
    current_S = S_0 # S[0]
    min_cost = C_0

    # Iterate from k=0 up to M-2
    # We have C[k] and S[k]
    # Calculate C[k+1] = C[k] + S[k]
    # Calculate S[k+1] = S[k] + DeltaS[k+1]

    for k in range(M - 1):
         # Update C[k+1]
         current_C += current_S
         min_cost = min(min_cost, current_C)

         # Update S[k+1] from S[k] using DeltaS[k+1]
         # S[k+1] = S[k] + DeltaS[k+1]
         next_k = k + 1
         delta_at_next_k = delta_S_values_arr[next_k]
         current_S += delta_at_next_k

    print(min_cost)


solve()