def can_fill_array(N, M, operations):
    # Try all possible combinations of operations (0,1,2) for each operation
    def try_operations(ops):
        array = [0] * N
        for i, (L, R) in enumerate(operations):
            if ops[i] == 0:
                continue
            elif ops[i] == 1:
                # Fill 1s from L to R
                for j in range(L-1, R):
                    array[j] = 1
            else:  # ops[i] == 2
                # Fill 1s outside L to R
                for j in range(N):
                    if j < L-1 or j > R-1:
                        array[j] = 1
        return all(x == 1 for x in array)

    def get_cost(ops):
        return sum(1 for x in ops if x != 0)

    # Try all possible combinations using dynamic programming
    def solve():
        min_cost = float('inf')
        best_ops = None
        
        # Start with trying just one operation
        for mask in range(3**M):
            ops = []
            temp = mask
            for _ in range(M):
                ops.append(temp % 3)
                temp //= 3
            
            if try_operations(ops):
                cost = get_cost(ops)
                if cost < min_cost:
                    min_cost = cost
                    best_ops = ops
        
        return min_cost, best_ops

    min_cost, best_ops = solve()
    
    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)
        print(*best_ops)

# Read input
N, M = map(int, input().split())
operations = []
for _ in range(M):
    L, R = map(int, input().split())
    operations.append((L, R))

can_fill_array(N, M, operations)