def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    MOD = 998244353

    # ans[k]: total number of arithmetic subsequences of length k
    # we will output ans[1..N]
    ans = [0] * (N+1)
    # every single element is a length-1 arithmetic subseq
    ans[1] = N

    # mp[i] is a dict: diff -> list cnt where
    # cnt[l] = number of arithmetic subsequences of length l ending at index i with common difference diff
    mp = [dict() for _ in range(N)]

    for i in range(N):
        ai = A[i]
        # consider all previous j < i as possible predecessor
        for j in range(i):
            diff = ai - A[j]
            # Extend all sequences at j with this diff
            cnts_j = mp[j].get(diff)
            # ensure mp[i][diff] exists
            cnts_i = mp[i].get(diff)
            if cnts_i is None:
                cnts_i = [0] * (N+1)
                mp[i][diff] = cnts_i

            # First, extend existing sequences of length >=2
            if cnts_j is not None:
                # cnts_j[l] is number of seqs of length l at j, we extend to length l+1
                # l runs from 2..N-1 max
                for l in range(2, N):
                    c = cnts_j[l]
                    if c:
                        nlen = l + 1
                        cnts_i[nlen] = (cnts_i[nlen] + c) % MOD
                        ans[nlen] = (ans[nlen] + c) % MOD

            # Second, the pair (j,i) itself is a new arithmetic subsequence of length 2
            cnts_i[2] = (cnts_i[2] + 1) % MOD
            ans[2] = (ans[2] + 1) % MOD

    # Print answers for lengths 1..N
    print(" ".join(str(ans[k] % MOD) for k in range(1, N+1)))

if __name__ == "__main__":
    main()