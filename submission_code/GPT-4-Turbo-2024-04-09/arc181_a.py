import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        P = list(map(int, data[index:index+N]))
        index += N
        
        # Check if already sorted
        if all(P[i] == i + 1 for i in range(N)):
            results.append(0)
            continue
        
        # Check the segments that are out of order
        out_of_order = 0
        if P[0] != 1:
            out_of_order += 1
        if P[-1] != N:
            out_of_order += 1
        
        # Check for internal segments
        internal_disorder = False
        for i in range(1, N):
            if P[i] != i + 1 and P[i-1] == i:
                internal_disorder = True
                break
        
        if out_of_order == 2 or internal_disorder:
            results.append(2)
        else:
            results.append(1)
    
    # Output results
    sys.stdout.write("
".join(map(str, results)) + "
")