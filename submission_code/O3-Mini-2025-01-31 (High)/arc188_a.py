def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read input
    N = int(data[0])
    K = int(data[1])
    S = data[2].strip()
    
    mod = 998244353
    # mapping letter -> mask value (we use bit0 for A, bit1 for B, bit2 for C)
    letter_to_mask = {'A': 1, 'B': 2, 'C': 4}
    # The 8 states (0..7) come in complementary pairs.
    # For 3–bit numbers these pairs are: {0,7}, {1,6}, {2,5}, {3,4}.
    # We assign a pair index to each state by taking the smaller of s and s^7.
    pair_idx = [0, 1, 2, 3, 3, 2, 1, 0]
    
    # We now set up our DP.
    # Each DP state will be a tuple (cur_state, k0, k1, k2, k3)
    # where cur_state is in 0..7 and k0,k1,k2,k3 count how many times a transition ended in
    # a state belonging to pair 0,1,2,3 respectively.
    # Note: our initial prefix P[0] is fixed and equals 0.
    # Since 0 is in pair 0, we “imagine” that the frequency of pair 0 is already 1.
    # (We do not store that extra 1 in the DP state – we just remember it when computing the final score.)
    dp = {}
    dp[(0, 0, 0, 0, 0)] = 1
    
    # Process the string S, one letter at a time.
    for ch in S:
        ndp = {}
        # if the letter is unknown we sum over A, B, C; otherwise a singleton.
        if ch == '?':
            allowed = ('A', 'B', 'C')
        else:
            allowed = (ch,)
        # For each DP state, process each allowed letter.
        for key, ways in dp.items():
            cur_state, k0, k1, k2, k3 = key
            for L in allowed:
                m = letter_to_mask[L]
                new_state = cur_state ^ m
                j = pair_idx[new_state]
                # update the count for pair j
                if j == 0:
                    newKey = (new_state, k0+1, k1, k2, k3)
                elif j == 1:
                    newKey = (new_state, k0, k1+1, k2, k3)
                elif j == 2:
                    newKey = (new_state, k0, k1, k2+1, k3)
                else:  # j == 3
                    newKey = (new_state, k0, k1, k2, k3+1)
                ndp[newKey] = (ndp.get(newKey, 0) + ways) % mod
        dp = ndp

    ans = 0
    # Now, each completed DP state corresponds to one full assignment of length N.
    # In a state (s, k0, k1, k2, k3), the final frequency in pair 0 is (k0+1) (remember the initial prefix)
    # and for j>=1 the frequency is just k_j.
    # Thus the total number of good substrings is:
    #    score = C(k0+1,2) + C(k1,2) + C(k2,2) + C(k3,2)
    # (Here C(n,2) = n*(n-1)//2.) We need to count only those assignments with score >= K.
    for key, ways in dp.items():
        _, k0, k1, k2, k3 = key
        # compute score per pair; note that for pair0 we use (k0+1) because of the initial prefix.
        sc = ((k0+1)*k0) // 2
        sc += (k1*(k1-1)) // 2
        sc += (k2*(k2-1)) // 2
        sc += (k3*(k3-1)) // 2
        if sc >= K:
            ans = (ans + ways) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()