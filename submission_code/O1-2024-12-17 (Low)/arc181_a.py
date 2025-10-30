def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    # Precompute triangular numbers for quick sum-check:
    # tri[n] = 1 + 2 + ... + n
    # We'll need up to n = 2 * 10^5
    maxN = 2 * 10**5
    tri = [0] * (maxN + 1)
    for i in range(1, maxN + 1):
        tri[i] = tri[i - 1] + i
    
    out = []
    
    for _ in range(t):
        N = int(input_data[idx]); idx += 1
        P = list(map(int, input_data[idx:idx+N]))
        idx += N
        
        # Check if already sorted:
        if all(P[i] == i+1 for i in range(N)):
            out.append('0')
            continue
        
        # We'll check if we can do it in 1 operation:
        # Condition for 1 operation with a chosen k:
        #  1) P[k-1] == k
        #  2) {P[0..k-2]} is exactly {1, 2, ..., k-1} (as a set)
        #  3) {P[k..N-1]} is exactly {k+1, k+2, ..., N} (as a set)
        #
        # We will build:
        #   prefix_good[i] = True iff P[0..i-1] is a permutation of {1..(i-1)}
        #   suffix_good[i] = True iff P[i-1..N-1] is a permutation of {i..N}
        # for i in [0..N+1], where prefix_good[0] and suffix_good[N+1] are trivially True.
        
        # Build prefix_good:
        prefix_good = [False] * (N+1)
        prefix_good[0] = True  # empty prefix is "good" for i=0
        freq = [0]*(N+1)  # frequency array for 1..N
        current_sum = 0
        distinct_count = 0
        
        for i in range(1, N+1):
            val = P[i-1]
            if 1 <= val <= i:
                if freq[val] == 0:
                    distinct_count += 1
                freq[val] += 1
                current_sum += val
                # We want the prefix P[0..i-1] to be {1..i}
                # so distinct_count = i, current_sum = tri[i]
                # and no duplicates
                if distinct_count == i and current_sum == tri[i]:
                    prefix_good[i] = True
            else:
                # If val is out of range for {1..i}, it can never be good here
                pass
        
        # Build suffix_good:
        suffix_good = [False] * (N+2)
        suffix_good[N+1] = True  # empty suffix is "good" for i=N+1
        freq = [0]*(N+1)
        current_sum = 0
        distinct_count = 0
        # We go from right to left
        # suffix_good[i] means P[i-1..N-1] is a permutation of {i..N} (size = N-i+1)
        # So if i = N, that suffix is just P[N-1..N-1], which should be {N}.
        # We'll build in decreasing order of i.
        
        for i in range(N, 0, -1):
            val = P[i-1]
            if i <= val <= N:
                if freq[val] == 0:
                    distinct_count += 1
                freq[val] += 1
                current_sum += val
                needed_count = N - i + 1
                needed_sum = tri[N] - tri[i-1]  # sum of i..N
                if distinct_count == needed_count and current_sum == needed_sum:
                    suffix_good[i] = True
            else:
                pass
        
        # Now check if there is a k in [1..N] for which:
        #  prefix_good[k-1] == True
        #  suffix_good[k+1] == True
        #  P[k-1] == k
        # If found => 1 operation, otherwise => 2
        can_do_in_one = False
        for k in range(1, N+1):
            if P[k-1] == k and prefix_good[k-1] and suffix_good[k+1]:
                can_do_in_one = True
                break
        
        if can_do_in_one:
            out.append('1')
        else:
            out.append('2')
    
    print('
'.join(out))