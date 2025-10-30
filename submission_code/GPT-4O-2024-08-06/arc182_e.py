def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    C = int(data[2])
    K = int(data[3])
    A = list(map(int, data[4:]))
    
    # Precompute A_i % M for all i
    A_mod = [a % M for a in A]
    
    # Calculate the minimum for each k in a single cycle (0 to M-1)
    min_values = []
    for k in range(M):
        min_value = float('inf')
        for a_mod in A_mod:
            value = (C * k + a_mod) % M
            if value < min_value:
                min_value = value
        min_values.append(min_value)
    
    # Sum of minimum values for one full cycle
    cycle_sum = sum(min_values)
    
    # Number of complete cycles
    full_cycles = K // M
    # Remaining k values after full cycles
    remaining_k = K % M
    
    # Total sum
    total_sum = full_cycles * cycle_sum + sum(min_values[:remaining_k])
    
    print(total_sum)