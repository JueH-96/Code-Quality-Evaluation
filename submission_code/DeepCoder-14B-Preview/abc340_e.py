def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    
    B = list(map(int, data[idx:idx+M]))
    
    current_A = A.copy()
    diff = [0] * (N + 1)  # 0..N indices
    
    for b in B:
        K = current_A[b]
        if K == 0:
            continue
        current_A[b] = 0
        
        q = K // N
        r = K % N
        
        # Add q to all elements
        diff[0] += q
        if N < len(diff):
            diff[N] -= q
        
        if r > 0:
            start = (b + 1) % N
            end = (b + r) % N
            
            if start <= end:
                diff[start] += 1
                if end + 1 < N:
                    diff[end + 1] -= 1
            else:
                diff[start] += 1
                diff[N] -= 1
                diff[0] += 1
                if end + 1 < N:
                    diff[end + 1] -= 1
    
    # Compute the additions
    additions = [0] * N
    current = 0
    for j in range(N):
        current += diff[j]
        additions[j] = current
    
    # Update current_A
    for j in range(N):
        current_A[j] += additions[j]
    
    print(' '.join(map(str, current_A)))

if __name__ == '__main__':
    main()