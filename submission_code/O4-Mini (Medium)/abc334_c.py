import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    A = set()
    for i in range(K):
        A.add(int(data[2 + i]))
    # Build sorted list S of remaining socks' colors
    # For each color i from 1..N, count = 1 if lost one, else 2
    S = []
    # total items M = 2N - K
    for i in range(1, N+1):
        if i in A:
            S.append(i)
        else:
            S.append(i)
            S.append(i)
    M = len(S)
    # If no pairs can be made or trivial, answer 0
    if M <= 1:
        print(0)
        return
    # Build array of differences between consecutive socks
    # D[j] = S[j+1] - S[j], j = 0..M-2
    D = [0] * (M-1)
    for j in range(M-1):
        D[j] = S[j+1] - S[j]
    # If M even: greedy pairing sum D at even indices
    if M % 2 == 0:
        total = 0
        # sum D[0] + D[2] + ... + D[M-2]
        for j in range(0, M-1, 2):
            total += D[j]
        print(total)
        return
    # M is odd: we must leave exactly one sock unpaired.
    # Precompute prefix sums of even-indexed D and suffix sums of odd-indexed D
    # prefix_even[x] = sum of D[i] for i < x and i even
    # length M+1 so indices 0..M
    prefix_even = [0] * (M+1)
    for j in range(M-1):
        prefix_even[j+1] = prefix_even[j] + (D[j] if (j & 1) == 0 else 0)
    prefix_even[M] = prefix_even[M-1]
    # suffix_odd[j] = sum of D[i] for i >= j and i odd
    # define length M+1, suffix_odd[M] = 0
    suffix_odd = [0] * (M+1)
    for j in range(M-1, -1, -1):
        # start from D index j if j<=M-2, else no D
        if j <= M-2 and (j & 1) == 1:
            suffix_odd[j] = suffix_odd[j+1] + D[j]
        else:
            suffix_odd[j] = suffix_odd[j+1]
    # Now try removing each sock at position k in 0..M-1
    # cost(k) = prefix_even[k-1] + suffix_odd[k+1] + (newgap if applies)
    # newgap = S[k+1] - S[k-1] if k-1>=0, k+1<M and (k-1)%2==0
    best = 10**30
    # We'll treat prefix_even[-1] as 0 by checking k>=1
    for k in range(M):
        # sum of even D before the gap k-1: that's prefix_even[k-1]
        if k >= 1:
            pe = prefix_even[k-1]
        else:
            pe = 0
        # sum of odd D after gap k: suffix_odd[k+1]
        so = suffix_odd[k+1] if (k+1) <= M else 0
        cost = pe + so
        # if we knit together S[k-1] and S[k+1], their gap sits at new index k-1
        # and is used if (k-1) is even
        if 0 < k < M-1 and ((k-1) & 1) == 0:
            # new gap cost
            cost += S[k+1] - S[k-1]
        if cost < best:
            best = cost
    print(best)

if __name__ == "__main__":
    main()