import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    A = list(map(int, data[index:index + N]))
    index += N
    
    M = int(data[index])
    index += 1
    B = list(map(int, data[index:index + M]))
    index += M
    
    L = int(data[index])
    index += 1
    C = list(map(int, data[index:index + L]))
    index += L
    
    Q = int(data[index])
    index += 1
    X = list(map(int, data[index:index + Q]))
    index += Q
    
    # Precompute possible sums of pairs from A and B
    AB_sums = set()
    for a in A:
        for b in B:
            AB_sums.add(a + b)
    
    # For each query, check if there exists a c in C such that (x - c) is in AB_sums
    results = []
    for x in X:
        possible = False
        for c in C:
            if (x - c) in AB_sums:
                possible = True
                break
        results.append("Yes" if possible else "No")
    
    # Print results
    print("
".join(results))