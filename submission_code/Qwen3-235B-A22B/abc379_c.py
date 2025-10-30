import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    X = list(map(int, input[idx:idx+M]))
    idx += M
    A = list(map(int, input[idx:idx+M]))
    idx += M
    
    sumA = sum(A)
    if sumA != N:
        print(-1)
        return
    
    # Sort the X and A in increasing order of X
    zip_XA = sorted(zip(X, A))
    X_sorted = [x for x, a in zip_XA]
    A_sorted = [a for x, a in zip_XA]
    
    if X_sorted[0] != 1:
        print(-1)
        return
    
    # Compute prefix sums
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i+1] = prefix[i] + A_sorted[i]
    
    # Check conditions for each X[i]
    for i in range(1, M):
        required = prefix[i] + 1
        if X_sorted[i] > required:
            print(-1)
            return
    
    # Compute the total steps
    total = 0
    for i in range(M):
        a = A_sorted[i]
        start = prefix[i] + 1
        end = prefix[i+1]
        # Calculate sum of x from start to end inclusive
        sum_x = (start + end) * a // 2
        contribution = sum_x - X_sorted[i] * a
        total += contribution
    
    print(total)

if __name__ == "__main__":
    main()