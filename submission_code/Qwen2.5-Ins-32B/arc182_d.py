import sys

def min_operations_to_match(A, B, M):
    N = len(A)
    operations = 0
    
    for i in range(N):
        diff = (B[i] - A[i]) % M
        if i > 0:
            if A[i] == A[i-1]:
                if B[i] == B[i-1]:
                    return -1
                else:
                    if (B[i] - A[i-1]) % M == 1:
                        diff = (M - 1) - diff
                    elif (B[i] - A[i-1]) % M == M - 1:
                        diff = (M - 1) + diff
            else:
                if B[i] == B[i-1]:
                    if (B[i] - A[i]) % M == 1:
                        diff = (M - 1) - diff
                    elif (B[i] - A[i]) % M == M - 1:
                        diff = (M - 1) + diff
        operations += min(diff, M - diff)
        
    return operations

def main():
    input = sys.stdin.read
    data = input().split()
    
    N, M = int(data[0]), int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:2*N+2]))
    
    result = min_operations_to_match(A, B, M)
    print(result)

if __name__ == "__main__":
    main()