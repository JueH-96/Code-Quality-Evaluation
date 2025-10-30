import sys
from itertools import combinations

def solve():
    N, K_orig = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    K_eff = K_orig
    use_complement_logic = False
    s_total = 0 # XOR sum of all elements in A

    # Determine if we should iterate K_orig elements or N-K_orig elements
    # K_eff will be the number of elements in combinations we iterate.
    # If K_orig > N/2, we iterate N-K_orig elements (elements to exclude).
    # The XOR sum of K_orig chosen elements = (XOR sum of all elements) ^ (XOR sum of (N-K_orig) excluded elements).
    if K_orig > N / 2:
        use_complement_logic = True
        # Calculate s_total (XOR sum of all elements in A)
        # This is only needed if use_complement_logic is True.
        for x_val in A:
            s_total ^= x_val
        K_eff = N - K_orig
    
    # Handle the case where K_eff is 0.
    # This happens if K_orig = N (since 1 <= K_orig <= N, K_orig cannot be 0).
    # If K_orig = N, then K_orig > N/2 is true (for N>=1), so use_complement_logic is true.
    # K_eff becomes N - N = 0.
    # We are choosing K_eff=0 elements to exclude. Their XOR sum is 0.
    # The result is s_total ^ 0 = s_total.
    if K_eff == 0:
        # This path implies K_orig = N.
        # s_total would have been computed because K_orig > N/2 (for N>=1).
        print(s_total)
        return

    max_val_found = 0 
    # All A_i are non-negative, so XOR sums are non-negative.
    # 0 is a safe initial value for max_val_found. It will be correctly updated
    # by the first combination's value, or remain 0 if all XOR sums are 0.

    for combo in combinations(A, K_eff):
        current_combo_xor = 0
        for x_in_combo in combo:
            current_combo_xor ^= x_in_combo
        
        val_to_consider = 0
        if use_complement_logic:
            # combo contains K_eff elements to exclude.
            # XOR sum of chosen elements = s_total ^ current_combo_xor
            val_to_consider = s_total ^ current_combo_xor
        else:
            # combo contains K_eff (i.e., K_orig) elements to include.
            val_to_consider = current_combo_xor
        
        max_val_found = max(max_val_found, val_to_consider)
        
    print(max_val_found)

if __name__ == '__main__':
    solve()