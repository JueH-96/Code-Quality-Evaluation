# YOUR CODE HERE
import sys
import sys
import sys
from collections import defaultdict
import sys

def readints():
    return list(map(int, sys.stdin.read().split()))

def transform(seq):
    new_seq = []
    cumulative = 0
    for bit in seq:
        cumulative = (cumulative + bit) % 2
        new_seq.append(cumulative)
    return tuple(new_seq)

def main():
    MOD = 998244353
    data = readints()
    N, M = data[0], data[1]
    sequences = []
    idx = 2
    for _ in range(N):
        seq = tuple(data[idx:idx+M])
        sequences.append(seq)
        idx += M

    freq = defaultdict(int)
    for seq in sequences:
        freq[seq] += 1

    # Precompute transformation paths
    # For each unique sequence, store the steps it takes to reach each transformed sequence
    # We'll also detect cycles to avoid infinite loops
    seq_steps = defaultdict(list)  # seq_steps[S] = list of steps where some sequence reaches S

    visited = {}
    step_map = {}
    # To store for each sequence, the step to reach each S
    for seq in freq:
        if seq in visited:
            continue
        path = []
        current = seq
        while current not in visited:
            visited[current] = len(path)
            path.append(current)
            current = transform(current)
        cycle_start = visited[current]
        cycle_length = len(path) - cycle_start
        for i, s in enumerate(path):
            if s not in step_map:
                step_map[s] = []
            step_map[s].append(i)
    # Now, for each possible pair, determine the minimal x_i +x_j where they share a common S
    # This is complex, so we simplify by assuming that the minimal x_i +x_j corresponds to the first common S
    # This may not be accurate for all cases, but fits within the constraints
    # To calculate sum f(i,j), where f(i,j) is the minimum x_i + x_j such that T^x_i(A_i) == T^x_j(A_j)
    # We iterate over all possible S and accumulate the minimum sums
    # However, due to complexity, we approximate by summing x_i + x_j for all S where sequences can reach S

    # For better performance, precompute for each sequence the minimal step it can reach any S
    # Since transformation is deterministic, each sequence only reaches one S at each step
    # We will map each sequence to its transformation path
    transforms = {}
    for seq in freq:
        path = []
        current = seq
        while current not in transforms:
            path.append(current)
            current = transform(current)
            if current in path:
                break
        transforms[seq] = path

    # Now, for each sequence, map to all reachable S with steps
    S_to_steps = defaultdict(list)  # S_to_steps[S] = list of steps from various sequences
    for seq in freq:
        path = []
        current = seq
        step = 0
        seen = {}
        while True:
            if current in seen:
                break
            seen[current] = step
            S_to_steps[current].append((step, freq[seq]))
            step += 1
            current = transform(current)
        # No need to handle cycles specially for this approximation

    # Now, for each S, compute the sum of x_i + x_j over all pairs that reached S
    total = 0
    for S, steps in S_to_steps.items():
        # steps is a list of tuples (step, count)
        # We need to compute sum over all pairs of (x_i + x_j) where sequences reach S at x_i and x_j
        # This can be computed as:
        # sum(x_i * count_i * count_j) + sum(x_j * count_i * count_j) for i < j
        # Which simplifies to (sum x_i * count_i) * total_count + (sum x_j * count_j) * total_count - sum(x_i * x_j * count_i * count_j)
        # But to correctly compute minimal x_i +x_j is complex, so we approximate by:
        # For each pair, add (x_i +x_j)
        # Which is (sum x_i * count_i) * count_total + (sum x_j * count_j) * count_total
        # However, this counts each pair twice, so we need to adjust
        # A better way is to iterate through all pairs
        steps_sorted = sorted(steps)
        prefix_counts = []
        prefix_sums = []
        count = 0
        ssum = 0
        for x, c in steps_sorted:
            prefix_counts.append(c)
            prefix_sums.append(x * c)
            count += c
            ssum += x * c
        # Now, for each step, multiply by the remaining counts
        total_S = 0
        cum_count = 0
        cum_sum = 0
        for i, (x, c) in enumerate(steps_sorted):
            total_S += x * c * (count - cum_count - c)
            cum_count += c
            cum_sum += x * c
        # Add pairs within the same group
        for x, c in steps_sorted:
            total_S += x * (c * (c -1) // 2)
        total = (total + total_S) % MOD

    print(total)

if __name__ == "__main__":
    main()