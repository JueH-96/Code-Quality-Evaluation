MOD = 998244353

N, Q = map(int, input().split())
operations = []
for _ in range(Q):
    p, v = map(int, input().split())
    operations.append((p-1, v))  # Convert to 0-indexed

# Memoization dictionary
memo = {}

def solve(op_idx, seq):
    if op_idx == Q:
        return 1
    
    state = (op_idx, seq)
    if state in memo:
        return memo[state]
    
    p, v = operations[op_idx]
    result = 0
    
    # Option 1: replace positions 0 to p with v
    can_do_1 = all(seq[i] <= v for i in range(p + 1))
    if can_do_1:
        new_seq = list(seq)
        for i in range(p + 1):
            new_seq[i] = v
        result = (result + solve(op_idx + 1, tuple(new_seq))) % MOD
    
    # Option 2: replace positions p to N-1 with v
    can_do_2 = all(seq[i] <= v for i in range(p, N))
    if can_do_2:
        new_seq = list(seq)
        for i in range(p, N):
            new_seq[i] = v
        result = (result + solve(op_idx + 1, tuple(new_seq))) % MOD
    
    memo[state] = result
    return result

initial_seq = tuple([0] * N)
print(solve(0, initial_seq))