import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Separate elements based on the parity of their original 1-based index
    # Original 1-based index i corresponds to 0-based index i-1.
    # Odd 1-based index i means i-1 is even (0, 2, 4, ...)
    # Even 1-based index i means i-1 is odd (1, 3, 5, ...)
    
    ovals = [] # Values from odd original indices (A[0], A[2], ...)
    evals = [] # Values from even original indices (A[1], A[3], ...)

    for i in range(N):
        if (i + 1) % 2 == 1: # Odd original index
            ovals.append(A[i])
        else: # Even original index
            evals.append(A[i])

    # Sort the values
    ovals.sort()
    evals.sort()

    total_score = 0

    if N % 2 == 0:
        # N is even, |ovals| = N/2, |evals| = N/2
        n = N // 2
        # To maximize sum |o - e|, pair smallest ovals with largest evals, etc.
        # Ovals sorted ascending: o_0, ..., o_{n-1}
        # Evals sorted ascending: e_0, ..., e_{n-1}
        # Optimal pairing is (o_k, e_{n-1-k})
        for k in range(n):
            total_score += abs(ovals[k] - evals[n - 1 - k])
    else:
        # N is odd, |ovals| = (N+1)/2, |evals| = (N-1)/2
        m = (N + 1) // 2 # size of ovals
        n = (N - 1) // 2 # size of evals
        
        # Ovals sorted ascending: o_0, ..., o_n
        # Evals sorted ascending: e_0, ..., e_{n-1}
        
        # We must use all evals (size n). We choose n elements from ovals (size n+1)
        # One element from ovals is left out. Let's iterate through which one is left out.
        # If ovals[j] is left out (0 <= j <= n), the remaining ovals are Ovals \ {ovals[j]}.
        # Let O'' be the remaining n ovals sorted ascending: o''_0, ..., o''_{n-1}.
        # The max score for this set is sum_{k=0}^{n-1} |evals[k] - o''_{n-1-k}|.
        # o''_{k} = ovals[k] if k < j, o''_{k} = ovals[k+1] if k >= j (using 0-based index for ovals list)
        # The element o''_{n-1-k} is used in the sum. Its original index in ovals (before removing ovals[j]) is:
        # If n-1-k < j  => o''_{n-1-k} = ovals[n-1-k]. This happens when k > n-1-j.
        # If n-1-k >= j => o''_{n-1-k} = ovals[n-1-k+1]. This happens when k <= n-1-j.

        # Let's define T1[k] and T2[k] to help calculate sums efficiently.
        # For k=0...n-1:
        # T1[k] corresponds to pairing evals[k] with ovals[n-k]. (Indices n-k range from n down to 1)
        # T2[k] corresponds to pairing evals[k] with ovals[n-1-k]. (Indices n-1-k range from n-1 down to 0)
        
        T1 = [0] * n
        T2 = [0] * n
        for k in range(n):
            T1[k] = abs(evals[k] - ovals[n - k])
            T2[k] = abs(evals[k] - ovals[n - 1 - k])
            
        # Compute prefix sums for T1 and suffix sums for T2
        PrefixT1 = [0] * (n + 1)
        for i in range(1, n + 1):
            PrefixT1[i] = PrefixT1[i-1] + T1[i-1]
            
        SuffixT2 = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            SuffixT2[i] = SuffixT2[i+1] + T2[i]

        # When leaving ovals[j] (0 <= j <= n), the sum is:
        # sum_{k=0 to n-j-1} |evals[k] - ovals[n-k]| + sum_{k=n-j to n-1} |evals[k] - ovals[n-1-k]|
        # This is PrefixT1[n-j] + SuffixT2[n-j].
        # Let p = n-j. As j goes from 0 to n, p goes from n down to 0.
        # We need to calculate max(PrefixT1[p] + SuffixT2[p]) for p = 0, 1, ..., n.
        
        max_score = 0
        for p in range(n + 1):
            max_score = max(max_score, PrefixT1[p] + SuffixT2[p])
            
        total_score = max_score

    print(total_score)

solve()