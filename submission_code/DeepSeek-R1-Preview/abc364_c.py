import sys

def main():
    N, X, Y = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    dishes = list(zip(A, B))
    
    # Check K=1
    for a, b in dishes:
        if a > X or b > Y:
            print(1)
            return
    
    # Sort the dishes by A, then B
    sortedA = sorted(dishes, key=lambda x: (x[0], x[1]))
    # Sort the dishes by B, then A
    sortedB = sorted(dishes, key=lambda x: (x[1], x[0]))
    
    # Precompute for sortedA
    prefixA_A = [0] * (N + 1)
    prefixB_A = [0] * (N + 1)
    for i in range(1, N+1):
        prefixA_A[i] = prefixA_A[i-1] + sortedA[i-1][0]
        prefixB_A[i] = prefixB_A[i-1] + sortedA[i-1][1]
    
    maxA_suffix_A = [0] * (N + 1)
    maxB_suffix_A = [0] * (N + 1)
    current_maxA = 0
    current_maxB = 0
    for i in range(N-1, -1, -1):
        current_maxA = max(current_maxA, sortedA[i][0])
        current_maxB = max(current_maxB, sortedA[i][1])
        maxA_suffix_A[i] = current_maxA
        maxB_suffix_A[i] = current_maxB
    
    # Precompute for sortedB
    prefixA_B = [0] * (N + 1)
    prefixB_B = [0] * (N + 1)
    for i in range(1, N+1):
        prefixA_B[i] = prefixA_B[i-1] + sortedB[i-1][0]
        prefixB_B[i] = prefixB_B[i-1] + sortedB[i-1][1]
    
    maxA_suffix_B = [0] * (N + 1)
    maxB_suffix_B = [0] * (N + 1)
    current_maxA = 0
    current_maxB = 0
    for i in range(N-1, -1, -1):
        current_maxA = max(current_maxA, sortedB[i][0])
        current_maxB = max(current_maxB, sortedB[i][1])
        maxA_suffix_B[i] = current_maxA
        maxB_suffix_B[i] = current_maxB
    
    # Binary search for K in 2..N-1
    low = 2
    high = N - 1
    answer = N  # Default to N if nothing found
    
    while low <= high:
        mid = (low + high) // 2
        possible = False
        
        # Check sortedA case
        if mid - 1 <= N:
            sumA = prefixA_A[mid - 1]
            sumB = prefixB_A[mid - 1]
            if mid <= N:
                remA = maxA_suffix_A[mid]
                remB = maxB_suffix_A[mid]
                if (sumA + remA > X) or (sumB + remB > Y):
                    possible = True
        
        # Check sortedB case
        if not possible and mid - 1 <= N:
            sumA_b = prefixA_B[mid - 1]
            sumB_b = prefixB_B[mid - 1]
            if mid <= N:
                remA_b = maxA_suffix_B[mid]
                remB_b = maxB_suffix_B[mid]
                if (sumA_b + remA_b > X) or (sumB_b + remB_b > Y):
                    possible = True
        
        if possible:
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    
    if answer <= N - 1:
        print(answer)
    else:
        # Check if K=N is possible (sum all dishes)
        totalA = sum(A)
        totalB = sum(B)
        if totalA > X or totalB > Y:
            print(N)
        else:
            print(N)
    
if __name__ == '__main__':
    main()