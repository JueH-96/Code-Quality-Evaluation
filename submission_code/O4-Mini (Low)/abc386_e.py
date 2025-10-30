import sys
import threading

def main():
    import sys
    from itertools import combinations
    
    data = sys.stdin.read().split()
    N = int(data[0]); K = int(data[1])
    A = list(map(int, data[2:]))

    # Trivial cases
    if K == 1:
        print(max(A))
        return
    if K == N:
        xor_all = 0
        for x in A:
            xor_all ^= x
        print(xor_all)
        return

    # To minimize per‐combination work, if K > N//2, we can instead choose
    # the smaller complement size M = N - K, iterate all M‐subsets, compute
    # the XOR of the complement, and from that derive the XOR of the chosen K:
    #    XOR(chosen) = total_xor ^ XOR(complement)
    # This is only valid when M is small.
    # We branch on which of K and M is smaller.
    total_xor = 0
    for x in A:
        total_xor ^= x
    
    # Decide to iterate over subsets of size s = min(K, N-K)
    # and use formula if we picked complements.
    if K <= N - K:
        s = K
        use_complement = False
    else:
        s = N - K
        use_complement = True

    ans = 0
    # Pre-bind for speed
    lst = A
    tx = total_xor
    if not use_complement:
        # direct: XOR over each K-combination
        for comb in combinations(lst, s):
            x = 0
            for v in comb:
                x ^= v
            if x > ans:
                ans = x
    else:
        # iterate over all M=N-K subsets, compute complement XOR
        # then chosen_K_xor = total_xor ^ xor_of_subset
        for comb in combinations(lst, s):
            x = 0
            for v in comb:
                x ^= v
            cx = tx ^ x
            if cx > ans:
                ans = cx

    print(ans)

if __name__ == "__main__":
    main()