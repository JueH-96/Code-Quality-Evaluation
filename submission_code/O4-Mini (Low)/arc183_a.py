#!/usr/bin/env python3
import sys
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    N, K = map(int, data)
    # We want the floor((S+1)/2)-th lex sequence of the multiset
    # where each of 1..N appears K times.
    # Total S = (N*K)! / (K!)^N
    # We'll construct it greedily:
    #   At each position, try each candidate x in increasing order.
    #   Let cnt[x] be remaining count of x, total = sum cnt.
    #   Let cur = number of permutations of the current multiset.
    #   If we choose x next, the number of permutations of the rest is
    #       c = cur * cnt[x] // total
    #   If c < rank, subtract c from rank and try next x.
    #   Else fix x, update cur=c, rank stays, decrement cnt[x], total--, go on.
    #
    # We keep cur as a Python big int.
    from math import factorial
    total = N * K
    # Compute S = total! // (K!**N)
    # and rank = floor((S+1)//2)
    # We'll build factorial(total) and factorial(K)
    fact = [1] * (total + 1)
    for i in range(1, total+1):
        fact[i] = fact[i-1] * i
    fK = fact[K]
    S = fact[total] // (fK ** N)
    rank = (S + 1) // 2
    # initial current_count = S
    cur = S
    # counts
    cnt = [0] * (N+1)
    for i in range(1, N+1):
        cnt[i] = K
    rem = total
    out = []
    # build total positions
    for pos in range(total):
        # try x from 1..N
        for x in range(1, N+1):
            if cnt[x] == 0:
                continue
            # if we pick x next, number of perms of the rest:
            # c = cur * cnt[x] // rem
            # Note integer division exact since cur divisible by rem
            c = (cur * cnt[x]) // rem
            if c < rank:
                rank -= c
            else:
                # pick x
                out.append(str(x))
                # update cur, cnt, rem
                cur = c
                cnt[x] -= 1
                rem -= 1
                break
    # print
    print(" ".join(out))

if __name__ == "__main__":
    main()