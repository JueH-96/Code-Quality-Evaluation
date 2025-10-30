import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Step 1: Construct the full sorted list of all available sock colors
    available_socks = []
    a_set = set(A)
    
    for i in range(1, N + 1):
        if i in a_set:
            available_socks.append(i)
        else:
            available_socks.append(i)
            available_socks.append(i)

    M = len(available_socks)

    if M == 0:
        print(0)
        return

    # Step 2: Precompute prefix sums of adjacent pair weirdness
    # pref_even_sum[i]: sum of |S[j+1]-S[j]| for j even, 0 <= j < i. (Pairs: (S[0],S[1]), (S[2],S[3]), ..., (S[i-2],S[i-1]))
    # pref_odd_sum[i]: sum of |S[j+1]-S[j]| for j odd, 0 <= j < i. (Pairs: (S[1],S[2]), (S[3],S[4]), ..., (S[i-2],S[i-1]) if i-2 is odd)
    
    pref_even_sum = [0] * (M + 1)
    pref_odd_sum = [0] * (M + 1)

    for i in range(M): # i iterates from 0 to M-1
        # Copy previous values
        pref_even_sum[i+1] = pref_even_sum[i]
        pref_odd_sum[i+1] = pref_odd_sum[i]

        # If i is part of a pair (S[i], S[i+1])
        if i + 1 < M:
            if i % 2 == 0: # S[i] is at an even index
                pref_even_sum[i+1] += abs(available_socks[i+1] - available_socks[i])
            else: # S[i] is at an odd index
                pref_odd_sum[i+1] += abs(available_socks[i+1] - available_socks[i])

    # Step 3: Precompute suffix sums of adjacent pair weirdness
    # suff_even_sum[i]: sum of |S[j+1]-S[j]| for j even, i <= j < M-1
    # suff_odd_sum[i]: sum of |S[j+1]-S[j]| for j odd, i <= j < M-1

    suff_even_sum = [0] * (M + 1)
    suff_odd_sum = [0] * (M + 1)

    for i in range(M - 1, -1, -1): # i iterates from M-1 down to 0
        # Copy previous values
        suff_even_sum[i] = suff_even_sum[i+1]
        suff_odd_sum[i] = suff_odd_sum[i+1]

        # If i is part of a pair (S[i], S[i+1])
        if i + 1 < M:
            if i % 2 == 0: # S[i] is at an even index
                suff_even_sum[i] += abs(available_socks[i+1] - available_socks[i])
            else: # S[i] is at an odd index
                suff_odd_sum[i] += abs(available_socks[i+1] - available_socks[i])

    # Step 4: Calculate minimum total weirdness
    min_total_weirdness = float('inf')

    if M % 2 == 0: # All socks can be paired
        # Cost is simply the sum of all even-indexed adjacent pairs (S[0],S[1]), (S[2],S[3]), ...
        min_total_weirdness = pref_even_sum[M]
    else: # One sock must be left unpaired
        for k in range(M): # Try leaving sock S[k] unpaired
            current_weirdness = 0
            
            # Cost from prefix S[0...k-1]: Pairs are (S[0],S[1]), (S[2],S[3]), ...
            current_weirdness += pref_even_sum[k]

            # Cost from suffix S[k+1...M-1]:
            # The remaining elements in this suffix are effectively re-indexed starting from 0.
            # So the first element S[k+1] (in original list) is now at index 0 in the suffix.
            # If k is even: S[k+1] is at an odd original index. It becomes the first element (index 0) of the suffix.
            #   So, we need sum of original odd-indexed pairs starting from k+1.
            if k % 2 == 0: 
                current_weirdness += suff_odd_sum[k+1]
            # If k is odd: S[k+1] is at an even original index. It becomes the first element (index 0) of the suffix.
            #   So, we need sum of original even-indexed pairs starting from k+1.
            else: 
                current_weirdness += suff_even_sum[k+1]
            
            min_total_weirdness = min(min_total_weirdness, current_weirdness)

    print(min_total_weirdness)

solve()