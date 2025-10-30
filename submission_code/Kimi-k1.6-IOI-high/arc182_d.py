def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N
    
    total_steps = 0
    
    for i in range(N):
        a = A[i]
        b = B[i]
        if a == b:
            continue
        
        forbidden = []
        if i > 0:
            forbidden.append(B[i-1])
        if i < N-1:
            forbidden.append(A[i+1])
        
        valid_clockwise = True
        for fv in forbidden:
            if (a < b and a <= fv <= b) or (a > b and (fv >= a or fv <= b)):
                valid_clockwise = False
                break
        
        valid_counter = True
        for fv in forbidden:
            if (a > b and b <= fv <= a) or (a < b and (fv <= a or fv >= b)):
                valid_counter = False
                break
        
        options = []
        if valid_clockwise:
            options.append((b - a) % M)
        if valid_counter:
            options.append((a - b) % M)
        
        if not options:
            print(-1)
            return
        
        total_steps += min(options)
    
    print(total_steps)

if __name__ == "__main__":
    main()