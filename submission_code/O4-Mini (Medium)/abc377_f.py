import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))

    R = set()
    C = set()
    D1 = set()
    D2 = set()
    # Read piece positions
    for _ in range(M):
        a = int(next(it)); b = int(next(it))
        R.add(a)
        C.add(b)
        D1.add(a - b)
        D2.add(a + b)

    # Total free pairs of row & column (not attacked by same-row or same-col)
    r = len(R)
    c = len(C)
    total_free = (N - r) * (N - c)

    # Count of free pairs attacked by diag1 lines: A_count = |{(i,j) in XxY : i-j in D1}|
    A_count = 0
    # For each diag1 index k, count j so that (i=j+k,j) is in free rows & free cols
    for k in D1:
        # j must satisfy 1 <= j <= N and 1 <= j+k <= N
        L = 1 if 1-k < 1 else (1-k)
        if L < 1:
            L = 1
        # but simpler: L = max(1, 1-k)
        L = max(1, 1 - k)
        # U = min(N, N-k)
        U = min(N, N - k)
        if L > U:
            continue
        total_j = U - L + 1
        # Now remove j in C, and j = r0 - k for any r0 in R
        forbidden = set()
        # j in C
        for col in C:
            if L <= col <= U:
                forbidden.add(col)
        # j so that i=j+k == some row in R
        # for each row r0, j = r0 - k
        for r0 in R:
            j0 = r0 - k
            if L <= j0 <= U:
                forbidden.add(j0)
        A_count += total_j - len(forbidden)

    # Count of free pairs attacked by diag2 lines: B_count = |{(i,j) in XxY : i+j in D2}|
    B_count = 0
    for s in D2:
        # j must satisfy 1 <= j <= N and 1 <= s-j <= N => s-N <= j <= s-1
        L = max(1, s - N)
        U = min(N, s - 1)
        if L > U:
            continue
        total_j = U - L + 1
        forbidden = set()
        # j in C
        for col in C:
            if L <= col <= U:
                forbidden.add(col)
        # j so that i=s-j is in R => j = s - r0
        for r0 in R:
            j0 = s - r0
            if L <= j0 <= U:
                forbidden.add(j0)
        B_count += total_j - len(forbidden)

    # Count of free pairs counted in both A and B: |{(i,j) in XxY: i-j in D1 AND i+j in D2}|
    AB_count = 0
    # For each (k,s), the intersection (i,j) = ((k+s)//2, (s-k)//2) if parity matches
    # Check if it's in free rows & free cols
    # Only consider pairs where (k+s) is even
    # Iterate smaller set outer to possibly speed up
    # But in worst case both ~1000, so ~1e6 checks: acceptable
    for k in D1:
        for s in D2:
            # Need k+s even
            spk = s + k
            if spk & 1:
                continue
            i = spk // 2
            j = (s - k) // 2
            # Check bounds
            if i < 1 or i > N or j < 1 or j > N:
                continue
            # Must not be in attacked row/col => i not in R, j not in C
            if i in R or j in C:
                continue
            # This (i,j) contributes
            AB_count += 1

    # Inclusion-exclusion on diag constraints over free row-col pairs:
    # safe = total_free - |A| - |B| + |A∩B|
    result = total_free - A_count - B_count + AB_count

    # Output the result
    sys.stdout.write(str(result))

if __name__ == "__main__":
    main()