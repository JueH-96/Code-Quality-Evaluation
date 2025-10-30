def main():
    import sys
    from bisect import bisect_left, bisect_right
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # Each query is of the form (R, X) and we also keep its original index.
    queries = []
    for idx in range(q):
        R = int(next(it))
        X = int(next(it))
        queries.append((R, X, idx))
    
    # Sort queries by R so we can process them in lock-step with the prefix of A.
    queries.sort(key=lambda t: t[0])
    
    # We'll simulate the standard "patience sorting" algorithm.
    # Let tail[k] be the smallest possible ending element of an increasing subsequence of length k+1.
    # Then the length of the longest increasing subsequence of the current prefix is len(tail).
    # Moreover, for a given threshold X, the answer for that prefix is the largest k such that
    # tail[k-1] <= X. Since tail is strictly increasing, that equals bisect_right(tail, X).
    answers = [0] * q
    tail = []
    qi = 0  # Pointer for queries
    
    # Process the array A in order (which corresponds to increasing prefix lengths).
    for i in range(n):
        a = A[i]
        # For strictly increasing sequences, use bisect_left to find the correct position.
        pos = bisect_left(tail, a)
        if pos == len(tail):
            tail.append(a)
        else:
            tail[pos] = a
        
        # Process queries whose prefix length R equals i+1.
        while qi < q and queries[qi][0] == i + 1:
            # Answer is the maximum k so that tail[k-1] <= X, i.e.,
            # count = bisect_right(tail, queries[qi][1]).
            answers[queries[qi][2]] = bisect_right(tail, queries[qi][1])
            qi += 1
    
    sys.stdout.write("
".join(map(str, answers)))

if __name__ == '__main__':
    main()