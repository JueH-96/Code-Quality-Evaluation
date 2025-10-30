import sys
from collections import defaultdict

def main():
    N, K = map(int, sys.stdin.readline().split())
    functions = []
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        functions.append((A, B))
    
    # Generate candidate list using multiple criteria
    s = 200  # Number of top functions per criterion
    candidates = set()
    
    # Sort by A descending
    a_sorted = sorted(range(N), key=lambda x: -functions[x][0])
    candidates.update(a_sorted[:s])
    
    # Sort by B descending
    b_sorted = sorted(range(N), key=lambda x: -functions[x][1])
    candidates.update(b_sorted[:s])
    
    # Sort by (A + B) descending
    sum_sorted = sorted(range(N), key=lambda x: -(functions[x][0] + functions[x][1]))
    candidates.update(sum_sorted[:s])
    
    # Sort by (A * B) descending
    product_sorted = sorted(range(N), key=lambda x: -(functions[x][0] * functions[x][1]))
    candidates.update(product_sorted[:s])
    
    # Sort by B / max(A - 1, 1) descending
    def score5(i):
        A, B = functions[i]
        denominator = max(A - 1, 1)
        return -B / denominator  # Negative for ascending order
    
    score5_sorted = sorted(range(N), key=lambda x: score5(x))
    candidates.update(score5_sorted[:s])
    
    # Convert to list and sort for deterministic order
    candidates = list(candidates)
    candidates.sort()
    C = len(candidates)
    
    # Prepare candidate A and B values
    candidate_A = [functions[i][0] for i in candidates]
    candidate_B = [functions[i][1] for i in candidates]
    
    # Beam search with dynamic programming
    current_states = defaultdict(int)
    current_states[0] = 1  # Initial state: mask 0 (no functions used), value 1
    
    for _ in range(K):
        new_states = defaultdict(int)
        for mask in current_states:
            current_value = current_states[mask]
            for i in range(C):
                if not (mask & (1 << i)):
                    new_value = current_value * candidate_A[i] + candidate_B[i]
                    new_mask = mask | (1 << i)
                    if new_value > new_states[new_mask]:
                        new_states[new_mask] = new_value
        # Sort new states by value in descending order and keep top 3000
        sorted_new = sorted(new_states.items(), key=lambda x: -x[1])
        current_states = defaultdict(int)
        for mask, value in sorted_new[:3000]:
            current_states[mask] = value
    
    # Find the maximum value among all remaining states
    max_val = max(current_states.values()) if current_states else 0
    print(max_val)

if __name__ == '__main__':
    main()