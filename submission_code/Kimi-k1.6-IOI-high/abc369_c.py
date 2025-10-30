def count_arithmetic_subarrays():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if N == 1:
        print(1)
        return
    
    D = [A[i+1] - A[i] for i in range(N-1)]
    total = 0
    current_diff = D[0]
    run_length = 1
    
    for diff in D[1:]:
        if diff == current_diff:
            run_length += 1
        else:
            total += run_length * (run_length + 1) // 2
            current_diff = diff
            run_length = 1
    # Add the last run
    total += run_length * (run_length + 1) // 2
    
    print(total + N)

count_arithmetic_subarrays()