def solve():
    N, M = map(int, input().split())
    A = []
    for _ in range(N):
        A.append(list(map(int, input().split())))
    
    def apply_operation(seq):
        result = []
        prefix_sum = 0
        for bit in seq:
            prefix_sum += bit
            result.append(prefix_sum % 2)
        return result
    
    def compute_f(seq_i, target):
        if seq_i == target:
            return 0
        
        current = seq_i[:]
        visited = {tuple(current)}
        
        # Upper bound to detect cycles
        max_steps = min(2**M if M <= 20 else 100000, 100000)
        
        for step in range(1, max_steps + 1):
            current = apply_operation(current)
            if current == target:
                return step
            
            state = tuple(current)
            if state in visited:
                return 0  # Cycle detected, target not reachable
            visited.add(state)
        
        return 0  # Too many steps, assume target not reachable
    
    total = 0
    MOD = 998244353
    
    for i in range(N):
        for j in range(i, N):
            f_val = compute_f(A[i], A[j])
            total = (total + f_val) % MOD
    
    print(total)

solve()