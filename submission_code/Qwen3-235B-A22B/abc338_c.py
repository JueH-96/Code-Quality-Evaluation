def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    
    Q = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    # Compute X_max
    x_max_candidates = []
    for i in range(N):
        if A[i] > 0:
            x_max_candidates.append(Q[i] // A[i])
    if not x_max_candidates:
        print(0)
        return
    X_max = min(x_max_candidates)
    
    ans = 0
    for x in range(X_max + 1):
        remaining = [Q[i] - A[i] * x for i in range(N)]
        y_candidates = []
        for i in range(N):
            if B[i] == 0:
                continue
            y_i = remaining[i] // B[i]
            y_candidates.append(y_i)
        if not y_candidates:
            continue  # Shouldn't happen per problem constraints
        y = min(y_candidates)
        ans = max(ans, x + y)
    
    print(ans)

if __name__ == "__main__":
    main()