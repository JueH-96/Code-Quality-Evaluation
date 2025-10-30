def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N, M = int(input[ptr]), int(input[ptr+1])
    ptr +=2
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    if A == B:
        print(0)
        return
    
    # Check if any B[i] == A[i+1] for i < N-1
    for i in range(N-1):
        if B[i] == A[i+1]:
            print(-1)
            return
    
    total = 0
    
    for i in range(N):
        left_f = B[i-1] if i > 0 else None
        right_f = A[i+1] if i < N-1 else None
        
        F = []
        if left_f is not None:
            F.append(left_f)
        if right_f is not None:
            F.append(right_f)
        
        S = A[i]
        T = B[i]
        
        if S == T:
            continue
        
        # Check clockwise
        blocked_clockwise = False
        for f in F:
            if (S < T and (S < f < T)) or (S > T and (f > S or f < T)):
                blocked_clockwise = True
                break
        
        steps_clockwise = (T - S) % M if not blocked_clockwise else float('inf')
        
        # Check counter-clockwise
        blocked_counter = False
        for f in F:
            if (S > T and (T < f < S)) or (S < T and (f < S or f > T)):
                blocked_counter = True
                break
        
        steps_counter = (S - T) % M if not blocked_counter else float('inf')
        
        min_steps = min(steps_clockwise, steps_counter)
        if min_steps == float('inf'):
            print(-1)
            return
        total += min_steps
    
    print(total)

if __name__ == '__main__':
    main()