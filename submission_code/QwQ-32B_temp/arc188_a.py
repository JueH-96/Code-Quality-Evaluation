MOD = 998244353

def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    all_states = []
    for a in 0, 1:
        for b in 0, 1:
            for c in 0, 1:
                all_states.append((a, b, c))
    
    def get_index(a, b, c):
        return a * 4 + b * 2 + c
    
    prev_count = [0] * 8
    prev_count[0] = 1  # Initial state (0,0,0)
    
    prev_total_substrings = [0] * 8
    prev_total_counts = [0] * 8
    prev_total_counts[0] = 1  # Cumulative counts up to previous step (step 0)
    
    for i in range(N):
        current_char = S[i]
        possible_chars = ['A', 'B', 'C'] if current_char == '?' else [current_char]
        
        # Precompute delta for each new_state
        delta = [0] * 8
        for new_state_idx in range(8):
            a, b, c = all_states[new_state_idx]
            for prev_prev_state_idx in range(8):
                pa, pb, pc = all_states[prev_prev_state_idx]
                da = (a - pa) % 2
                db = (b - pb) % 2
                dc = (c - pc) % 2
                if da == db and db == dc:
                    delta[new_state_idx] = (delta[new_state_idx] + prev_total_counts[prev_prev_state_idx]) % MOD
        
        # Initialize new arrays
        new_count = [0] * 8
        new_total_substrings = [0] * 8
        for current_char_choice in possible_chars:
            for prev_state_idx in range(8):
                if prev_count[prev_state_idx] == 0:
                    continue
                pa, pb, pc = all_states[prev_state_idx]
                if current_char_choice == 'A':
                    na, nb, nc = (pa + 1) % 2, pb, pc
                elif current_char_choice == 'B':
                    na, nb, nc = pa, (pb + 1) % 2, pc
                else:  # 'C'
                    na, nb, nc = pa, pb, (pc + 1) % 2
                new_state_idx = get_index(na, nb, nc)
                # Update new_count
                new_count[new_state_idx] = (new_count[new_state_idx] + prev_count[prev_state_idx]) % MOD
                # Update contribution to new_total_substrings
                contrib = (prev_total_substrings[prev_state_idx] * prev_count[prev_state_idx]) % MOD
                new_total_substrings[new_state_idx] = (new_total_substrings[new_state_idx] + contrib) % MOD
        
        # Add delta contributions
        for s in range(8):
            delta_s = delta[s]
            cnt = new_count[s]
            new_total_substrings[s] = (new_total_substrings[s] + delta_s * cnt) % MOD
        
        # Compute new_total_counts
        new_total_counts = [0] * 8
        for s in range(8):
            new_total_counts[s] = (prev_total_counts[s] + new_count[s]) % MOD
        
        # Update previous arrays for next iteration
        prev_count, prev_total_substrings, prev_total_counts = new_count, new_total_substrings, new_total_counts
    
    # Calculate the answer
    ans = 0
    for s in range(8):
        if prev_total_substrings[s] >= K:
            ans = (ans + prev_count[s]) % MOD
    print(ans % MOD)

if __name__ == "__main__":
    main()