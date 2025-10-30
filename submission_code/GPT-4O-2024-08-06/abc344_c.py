# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    # Read N and A
    N = int(data[index])
    index += 1
    A = list(map(int, data[index:index + N]))
    index += N
    
    # Read M and B
    M = int(data[index])
    index += 1
    B = list(map(int, data[index:index + M]))
    index += M
    
    # Read L and C
    L = int(data[index])
    index += 1
    C = list(map(int, data[index:index + L]))
    index += L
    
    # Read Q and X
    Q = int(data[index])
    index += 1
    X = list(map(int, data[index:index + Q]))
    
    # Precompute all possible sums of A[i] + B[j]
    sumAB = set()
    for a in A:
        for b in B:
            sumAB.add(a + b)
    
    # For each query, check if there is a valid combination
    results = []
    for x in X:
        found = False
        for c in C:
            if (x - c) in sumAB:
                found = True
                break
        if found:
            results.append("Yes")
        else:
            results.append("No")
    
    # Print results
    for result in results:
        print(result)