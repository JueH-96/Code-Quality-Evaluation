import sys

def solve():
    N, A, B = map(int, sys.stdin.readline().split())
    D_values_str = sys.stdin.readline().split()

    L = A + B

    # Calculate D_i % L and store distinct sorted values in p_prime
    distinct_p_mod_L = set()
    for s_val in D_values_str:
        val = int(s_val)
        distinct_p_mod_L.add(val % L)
    
    # N >= 1, so distinct_p_mod_L is never empty.
    p_prime = sorted(list(distinct_p_mod_L))
    
    M = len(p_prime)
    
    # Iterate through each p_prime[k_idx].
    # Consider a shift that maps p_prime[k_idx] to day 0.
    # The holidays are days 0, 1, ..., A-1.
    # If p_prime[k_idx] is mapped to 0, all other plan days p_prime[j]
    # are mapped to (p_prime[j] - p_prime[k_idx] + L) % L.
    # The minimum of these shifted values is 0.
    # The maximum of these shifted values occurs for the point that was
    # cyclically just before p_prime[k_idx] in the sorted list.
    # This point is p_prime[(k_idx - 1 + M) % M].
    # Let this predecessor be pred_val. Its shifted value is (pred_val - p_prime[k_idx] + L) % L.
    # This is the "span" of the set of points when p_prime[k_idx] is aligned to 0.
    # This span must be <= A - 1 for all points to fit in [0, A-1].
    
    for k_idx in range(M):
        current_val_at_k_idx = p_prime[k_idx]
        
        # Determine the predecessor in the sorted circular list of p_prime values
        predecessor_idx = (k_idx - 1 + M) % M # Handles wrap-around for k_idx = 0
        predecessor_val = p_prime[predecessor_idx]
        
        # Calculate the span. This is the largest day number any plan falls on,
        # assuming the plan corresponding to current_val_at_k_idx falls on day 0.
        span = (predecessor_val - current_val_at_k_idx + L) % L
        
        if span <= A - 1:
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    solve()