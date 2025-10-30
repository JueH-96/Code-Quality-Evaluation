def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = list(map(int, data[1:N+1]))
    A = list(map(int, data[N+1:2*N+1]))
    B = list(map(int, data[2*N+1:3*N+1]))
    
    # Compute x_max
    x_max = float('inf')
    for i in range(N):
        if A[i] > 0:
            x_max = min(x_max, Q[i] // A[i])
    x_max = max(x_max, 0)
    
    # Function to compute y_max for a given x
    def y_max(x):
        y_candidates = []
        for i in range(N):
            if B[i] > 0:
                if Q[i] - x * A[i] >= 0:
                    y_candidates.append((Q[i] - x * A[i]) // B[i])
                else:
                    return -1  # Invalid x
            else:
                if A[i] > 0 and Q[i] - x * A[i] < 0:
                    return -1  # Invalid x
        return min(y_candidates)
    
    # Function S(x) = x + y_max(x)
    def S(x):
        y = y_max(x)
        if y == -1:
            return -1
        else:
            return x + y
    
    # Ternary search to find the x that maximizes S(x)
    def ternary_search(left, right, func):
        while right - left > 3:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            f1 = func(mid1)
            f2 = func(mid2)
            if f1 < f2:
                left = mid1
            else:
                right = mid2
        max_S = -1
        for x in range(left, right + 1):
            val = func(x)
            if val > max_S:
                max_S = val
        return max_S
    
    max_S = ternary_search(0, x_max, S)
    
    # If no valid combination is found, set S to 0
    if max_S == -1:
        print(0)
    else:
        print(max_S)

if __name__ == "__main__":
    main()