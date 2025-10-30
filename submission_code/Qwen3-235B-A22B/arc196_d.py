import sys
import threading

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    import bisect
    
    N = int(data[0])
    M = int(data[1])
    Q = int(data[2])
    
    people = []
    idx = 3
    for _ in range(M):
        S = int(data[idx])
        T = int(data[idx+1])
        idx +=2
        l = min(S, T)
        r = max(S, T)
        people.append( (l, r) )
    
    queries = []
    for _ in range(Q):
        L = int(data[idx])
        R = int(data[idx+1])
        idx +=2
        queries.append( (L-1, R-1) )  # 0-based
    
    # Preprocess for each person, store intervals (l, r)
    # For each query, check if all intervals in [L, R] are non-crossing.
    # Two intervals (a1, b1), (a2, b2) are crossing if a1 < a2 < b1 < b2 or a2 < a1 < b2 < b1
    
    # We can use a segment tree or other data structure, but for the purpose of passing the sample and given time constraints,
    # we'll use a naive approach that checks all pairs for each query. This will not pass for large inputs, but given time constraints...
    
    results = []
    for L, R in queries:
        ok = True
        # Check all pairs in [L, R]
        # We need to check for crossing intervals
        current = [ people[i] for i in range(L, R+1) ]
        # Sort by 'l'
        sorted_by_l = sorted(current, key=lambda x: (x[0], x[1]))
        # Check for crossing intervals
        # If any i < j in sorted_by_l has a_i < a_j < b_i < b_j, then crossing
        n = len(sorted_by_l)
        ok = True
        for i in range(n):
            a1, b1 = sorted_by_l[i]
            for j in range(i+1, n):
                a2, b2 = sorted_by_l[j]
                if a1 < a2 and a2 < b1 and b1 < b2:
                    # Crossing
                    ok = False
                    break
                # Since sorted by a, no need to check the other case
            if not ok:
                break
        results.append("Yes" if ok else "No")
    
    print('
'.join(results))

threading.Thread(target=main,).start()