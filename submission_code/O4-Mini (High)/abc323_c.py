def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(M)]
    S = [next(it) for _ in range(N)]
    # Compute each player's current total score (solved problems + bonus i)
    T = []
    for i in range(N):
        solved_sum = 0
        si = S[i]
        for j in range(M):
            if si[j] == 'o':
                solved_sum += A[j]
        T.append(solved_sum + (i+1))
    # For each player, find how many more problems needed
    out = []
    for i in range(N):
        # max other player's score
        mx = -1
        ti = T[i]
        for j in range(N):
            if j == i:
                continue
            if T[j] > mx:
                mx = T[j]
        # deficit to exceed mx
        need = mx - ti + 1
        if need <= 0:
            out.append("0")
            continue
        # gather unsolved problem scores
        unsolved = []
        si = S[i]
        for j in range(M):
            if si[j] == 'x':
                unsolved.append(A[j])
        unsolved.sort(reverse=True)
        s = 0
        cnt = 0
        for v in unsolved:
            s += v
            cnt += 1
            if s >= need:
                break
        out.append(str(cnt))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()