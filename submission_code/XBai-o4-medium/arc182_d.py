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
    
    if A == B:
        print(0)
        return
    
    left_to_right = True
    for i in range(N-1):
        if B[i] == A[i+1]:
            left_to_right = False
    
    right_to_left = True
    for i in range(1, N):
        if B[i] == A[i-1]:
            right_to_left = False
    
    if not left_to_right and not right_to_left:
        print(-1)
        return
    
    total = 0
    for a, b in zip(A, B):
        delta = (b - a) % M
        total += min(delta, M - delta)
    
    print(total)

if __name__ == "__main__":
    main()