def calculate_operations(N, H):
    A = [0] * (N + 1)
    results = []
    
    for i in range(1, N + 1):
        # We need to find the number of operations before A[i] > 0
        # A[i] > 0 will happen when A[0] has been incremented enough
        # and the operations have allowed A[i] to be incremented.
        
        # The number of operations needed to make A[i] > 0 is:
        # A[0] must be at least 1 (to increment A[0])
        # Then we need to ensure that A[i] can be incremented from A[i-1]
        # This means A[i-1] must be greater than H[i] and A[i-1] must be decremented
        # to allow A[i] to increment.
        
        # The number of operations needed to make A[i] > 0 is:
        # 1 (to increment A[0]) + H[i] (to allow A[i] to be incremented)
        
        # The total operations needed to make A[i] > 0 is:
        # operations = H[i] + i (because we need to increment A[0] i times)
        
        operations = H[i - 1] + i
        results.append(operations)
        
    return results

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N + 1]))
    
    results = calculate_operations(N, H)
    print(" ".join(map(str, results)))

if __name__ == "__main__":
    main()