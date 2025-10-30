def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    original_A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(lambda x: int(x), input[ptr:ptr+M]))
    
    sum_S = [0] * N
    sum_full = 0
    delta_remaining = [0] * N
    
    for b in B:
        current_B = b
        s_i = original_A[current_B] if current_B < N else 0  # Ensure index is valid
        sum_S[current_B] += s_i
        # Update A (though not needed for computation, but ensures correctness)
        if current_B < N:
            original_A[current_B] = 0
        
        full_cycles = s_i // N
        sum_full += full_cycles
        remaining = s_i % N
        
        if remaining > 0:
            start = (current_B + 1) % N
            end = (start + remaining - 1) % N
            
            if start <= end:
                delta_remaining[start] += 1
                if end + 1 < N:
                    delta_remaining[end + 1] -= 1
                else:
                    delta_remaining[0] -= 1
            else:
                delta_remaining[start] += 1
                delta_remaining[0] += 1
                delta_remaining[end + 1] -= 1
    
    # Compute prefix sums for delta_remaining
    delta_remaining_prefix = [0] * N
    current = 0
    for i in range(N):
        current += delta_remaining[i]
        delta_remaining_prefix[i] = current
    
    # Calculate the result
    result = []
    for i in range(N):
        res = (original_A[i] - sum_S[i]) + sum_full + delta_remaining_prefix[i]
        result.append(str(res))
    
    print(' '.join(result))

if __name__ == '__main__':
    main()