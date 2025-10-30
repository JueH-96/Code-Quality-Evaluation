# YOUR CODE HERE
import sys, bisect

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx += 1
    A = list(map(int, data[idx:idx+N])); idx += N
    Q = int(data[idx]); idx += 1
    queries = []
    for _ in range(Q):
        l = int(data[idx]); r = int(data[idx+1]); idx += 2
        queries.append((l, r))
    
    sleep_starts = A[1::2]
    sleep_ends = A[2::2]
    M = len(sleep_starts)
    
    prefix_sum_durations = [0] * (M+1)
    prefix_sum_S = [0] * (M+1)
    for i in range(M):
        duration = sleep_ends[i] - sleep_starts[i]
        prefix_sum_durations[i+1] = prefix_sum_durations[i] + duration
        prefix_sum_S[i+1] = prefix_sum_S[i] + sleep_starts[i]
    
    # E is sleep_ends sorted
    E = sleep_ends
    S = sleep_starts
    # They are already sorted because A is sorted
    
    def total_sleep_up_to(X):
        # m: number of E_i <= X
        m = bisect.bisect_right(E, X)
        # k: number of S_i < X
        k = bisect.bisect_left(S, X)
        # Compute prefix_sum_durations[m]
        sum_durations = prefix_sum_durations[m]
        # Compute (k - m) * X - (prefix_sum_S[k] - prefix_sum_S[m])
        overlap = (k - m) * X - (prefix_sum_S[k] - prefix_sum_S[m])
        return sum_durations + overlap if k > m else sum_durations
    
    for l, r in queries:
        sleep_r = total_sleep_up_to(r)
        sleep_l = total_sleep_up_to(l)
        print(sleep_r - sleep_l)

if __name__ == "__main__":
    main()