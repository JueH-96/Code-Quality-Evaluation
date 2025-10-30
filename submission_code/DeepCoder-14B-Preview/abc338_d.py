def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))
    
    if M == 1:
        print(0)
        return
    
    # Compute original_sum
    original_sum = 0
    for i in range(M-1):
        A = X[i]
        B = X[i+1]
        d_clock = (B - A) % N
        d_counter = (A - B) % N
        if d_clock < d_counter:
            original_sum += d_clock
        else:
            original_sum += d_counter
    
    # Initialize difference array
    diff = [0] * (N + 2)  # 0..N+1
    
    for i in range(M-1):
        A = X[i]
        B = X[i+1]
        d_clock = (B - A) % N
        d_counter = (A - B) % N
        
        if d_clock < d_counter:
            delta = d_counter - d_clock
            if B > A:
                L = A
                R = B - 1
                if L <= R:
                    diff[L] += delta
                    if R + 1 <= N:
                        diff[R + 1] -= delta
            else:
                # Add delta to [A, N]
                L = A
                R = N
                if L <= R:
                    diff[L] += delta
                    if R + 1 <= N:
                        diff[R + 1] -= delta
                # Add delta to [1, B-1]
                if B - 1 >= 1:
                    L = 1
                    R = B - 1
                    if L <= R:
                        diff[L] += delta
                        if R + 1 <= N:
                            diff[R + 1] -= delta
        elif d_counter < d_clock:
            delta = d_clock - d_counter
            if B < A:
                L = B
                R = A - 1
                if L <= R:
                    diff[L] += delta
                    if R + 1 <= N:
                        diff[R + 1] -= delta
            else:
                # Add delta to [B, N]
                L = B
                R = N
                if L <= R:
                    diff[L] += delta
                    if R + 1 <= N:
                        diff[R + 1] -= delta
                # Add delta to [1, A-1]
                if A - 1 >= 1:
                    L = 1
                    R = A - 1
                    if L <= R:
                        diff[L] += delta
                        if R + 1 <= N:
                            diff[R + 1] -= delta
    
    # Compute prefix sums to get sum_delta for each bridge
    current = 0
    sum_delta = [0] * (N + 1)  # 1-based indexing
    for i in range(1, N + 1):
        current += diff[i]
        sum_delta[i] = current
    
    # Find the minimal new_sum
    min_new_sum = float('inf')
    for b in range(1, N + 1):
        new_sum = original_sum + sum_delta[b]
        if new_sum < min_new_sum:
            min_new_sum = new_sum
    
    print(min_new_sum)

if __name__ == '__main__':
    main()