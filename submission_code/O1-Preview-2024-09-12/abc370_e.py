# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    mod = 998244353

    # Build prefix sums
    S = [0] * (N + 1)
    for i in range(N):
        S[i+1] = S[i] + A[i]

    from collections import defaultdict

    # Map cumulative sum to list of positions where sum occurs
    sum_to_positions = defaultdict(list)
    sum_to_positions[0].append(0)  # s = 0 at position 0

    prohibited_intervals = []
    for i in range(1, N+1):
        s = S[i]
        if (s - K) in sum_to_positions:
            for l in sum_to_positions[s - K]:
                start = l + 1  # positions are 1-indexed
                end = i  # positions are 1-indexed
                if start <= end:
                    # Prohibited interval over cuts positions from start to end -1
                    prohibited_intervals.append((start, end - 1))
        sum_to_positions[s].append(i)

    # Now merge overlapping prohibited intervals over cuts positions
    prohibited_intervals.sort()
    merged_intervals = []
    for interval in prohibited_intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last_start, last_end = merged_intervals[-1]
            current_start, current_end = interval
            if current_start <= last_end + 1:
                # Merge intervals
                merged_intervals[-1] = (last_start, max(last_end, current_end))
            else:
                merged_intervals.append(interval)

    total_cuts = N -1
    total_positions_in_intervals = 0
    result = 1
    for interval in merged_intervals:
        a, b = interval
        length = b - a +1
        if length <= 0:
            continue
        total_positions_in_intervals += length
        temp = pow(2, length, mod) - 1
        if temp < 0:
            temp += mod
        result = (result * temp) % mod

    positions_outside_intervals = total_cuts - total_positions_in_intervals
    total_ways_outside = pow(2, positions_outside_intervals, mod)
    result = (result * total_ways_outside) % mod
    print(result)

threading.Thread(target=main).start()