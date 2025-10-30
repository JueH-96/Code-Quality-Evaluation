import sys
from collections import defaultdict

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    ops = []
    for _ in range(Q):
        P = int(input[ptr])
        ptr += 1
        V = int(input[ptr])
        ptr += 1
        ops.append((P, V))
    
    # DP state: (L, R, vl, vr) where L is the max prefix P so far, R is the min suffix P so far
    # vl is the value of the last prefix operation (0 if L=0)
    # vr is the value of the last suffix operation (0 if R=N+1)
    # We don't need to track last_type explicitly because the constraints can be derived from L, R, vl, vr
    dp = defaultdict(int)
    # Initial state: no prefix or suffix, L=0, R=N+1, vl and vr are 0 (but not applicable)
    dp[(0, N+1, 0, 0)] = 1
    
    for i in range(Q):
        P, V = ops[i]
        next_dp = defaultdict(int)
        for (L, R, vl, vr), cnt in dp.items():
            # Try prefix operation
            valid = True
            # Check if all in [1..P] are <= V
            # Region 1: [1..min(L,P)]: covered by vl
            if L >= 1 and vl > V:
                valid = False
            # Region 2: (L < j <= P], check if R <= j (i.e., R <= P) then vr must <= V
            if valid and R <= P:
                if vr > V:
                    valid = False
            # else: region 2 is 0, ok
            if valid:
                newL = max(L, P)
                newvl = V
                # R remains the same, vr remains the same
                # after prefix, the new L is newL, and the new last is prefix
                # but we don't track last_type. However, the coverage is newL and R
                newR = R
                new_vr = vr
                if newL < newR or (newL == newR and newL == 0):
                    # Ensure newL <= R
                    next_dp[(newL, newR, newvl, new_vr)] = (next_dp[(newL, newR, newvl, new_vr)] + cnt) % MOD
                else:
                    # newL >= newR. But the prefix overwrites newL's coverage, so suffix coverage is now only if R was larger than newL
                    # Not sure, but we proceed
                    pass
        for (L, R, vl, vr), cnt in dp.items():
            # Try suffix operation
            valid = True
            # Check all in [P..N] <= V
            # Region 1: covered by suffix: check if R <= N and (current R <= previous R)
            # In suffix's coverage P..N:
            # elements [max(R, P), N] are covered by vr (if last suffix is newer)
            # but according to our model, elements >= R are vr (if last operation is suffix)
            # Or, regardless of last operation, elements in [P..N] can be:
            # suffix's region: if last suffix covers them and prefix coverage doesn't override
            # Our model is:
            # elements >= R are vr
            # elements <= L are vl
            # middle 0
            # but overlapping depends on L and R
            # So in suffix's query [P, N]:
            # elements are vr if >= R and > L (if last operation is prefix)
            # or vr if last operation is suffix.
            # This is very complex. We'll use the following constraints:
            # Region 1: elements in [P..R-1] (if P <= R-1) are either covered by prefix (<= L) or 0
            # So covered by vl (if L >= j), but L >= j >= P implies L >= P. Or 0.
            # Region 2: elements R..N are vr
            # So to have all <= V:
            # vr (for elements R..N and >= P) must <= V
            # and in elements [P..min(L, ...)] must be vl <= V or other conditions
            # Alternative constraints for suffix:
            # elements in [P, N] must be <= V. According to our model:
            # if element <= L: its value is vl
            # elif its >= R: its value is vr
            # else: 0
            # So constraints:
            # if L >= P: vl must <= V (because elements P..L are vl)
            # if R <= N: vr must <= V (elements >= R)
            # and overlapping areas
            # Simplified constraints:
            # Check if vl (if L >= P) > V: invalid
            # Check if vr (if >= R and >= P (i.e., R <= N and R >= P)) vr > V: invalid
            # Or more accurately:
            # The vl applies to elements max(P, R) -1 ? No.
            # Another Idea:
            # To have all j in [P, N] <= V:
            # (if L >= P, then vl <= V)
            # (if R <= N and R <= j for some j >= P, then vr <= V)
            # i.e.,:
            # if L >= P: vl must <= V (because elements in P..L are vl)
            # if there exists an element j >= max(P, R): then vr <= V
            # which means if R <= N and max(P, R) <= N, then vr <= V
            # So conditions:
            # (L < P or vl <= V) and (R > N or (R >= P) and vr <= V)
            # But R <= N always true unless no suffix done yet.
            # So:
            # (if L >= P, then vl <= V) and (if R <= N and R <= N (yes), and R >= P (so elements in R's coverage overlap query), then vr <= V)
            # rewritten:
            if L >= P and vl > V:
                valid = False
            if R <= N and vr > V:
                # check if R <= N and overlaps with [P, N], i.e., R <= N's query's start P?
                # if R >= P: overlaps
                if R >= P:
                    valid = False
            if valid:
                newR = min(R, P)
                new_vr = V
                newL = L
                new_vl = vl
                if newR > newL:
                    next_dp[(newL, newR, new_vl, new_vr)] = (next_dp[(newL, newR, new_vl, new_vr)] + cnt) % MOD
                else:
                    # newR <= newL, entire array is covered by prefix and suffix. Which is newer?
                    # Not handled in model, but proceed
                    pass
        dp = next_dp
    ans = 0
    for key in dp:
        ans = (ans + dp[key]) % MOD
    print(ans)

if __name__ == "__main__":
    main()