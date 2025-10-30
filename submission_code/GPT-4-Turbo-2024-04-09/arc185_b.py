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
        A = list(map(int, data[index:index + N]))
        index += N
        
        possible = True
        # We need to check if we can make the array non-decreasing
        # by only moving "excess" from left to right.
        for i in range(N - 1):
            if A[i] > A[i + 1]:
                diff = A[i] - A[i + 1]
                A[i] -= diff
                A[i + 1] += diff
        
        # After processing, check if the array is non-decreasing
        for i in range(N - 1):
            if A[i] > A[i + 1]:
                possible = False
                break
        
        if possible:
            results.append("Yes")
        else:
            results.append("No")
    
    # Print all results for each test case
    sys.stdout.write("
".join(results) + "
")