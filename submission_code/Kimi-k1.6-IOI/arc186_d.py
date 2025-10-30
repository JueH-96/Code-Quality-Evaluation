MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    max_balance = 2 * N  # Sufficiently large to cover all possible balances
    tight = [0] * (max_balance + 2)
    not_tight = [0] * (max_balance + 2)
    current_tight = [0] * (max_balance + 2)
    current_not_tight = [0] * (max_balance + 2)
    
    # Initialize with balance 1, tight is True
    current_tight[1] += 1
    
    for i in range(N):
        # Reset the next state arrays
        for k in range(max_balance + 2):
            tight[k] = 0
            not_tight[k] = 0
        
        # Process tight states
        for b in range(max_balance + 1):
            cnt = current_tight[b]
            if cnt == 0:
                continue
            max_v = A[i]
            if i < N - 1:
                min_v = 1 if b == 1 else 0
            else:
                min_v = 0
            
            # Valid v range is [min_v, max_v]
            if min_v > max_v:
                continue
            
            # Handle v < A[i] (contribute to not_tight)
            lower_v = min_v
            upper_v = min(max_v, A[i] - 1)
            if lower_v <= upper_v:
                new_b_min = b - 1
                new_b_max = b - 1 + upper_v
                if new_b_min < 0:
                    new_b_min = 0
                if new_b_max > max_balance:
                    new_b_max = max_balance
                if new_b_min <= new_b_max:
                    not_tight[new_b_min] = (not_tight[new_b_min] + cnt) % MOD
                    if new_b_max + 1 <= max_balance:
                        not_tight[new_b_max + 1] = (not_tight[new_b_max + 1] - cnt) % MOD
            
            # Handle v == A[i] (contribute to tight)
            v = A[i]
            if v >= min_v and v <= max_v:
                new_b = b - 1 + v if v > 0 else b - 1
                if i == N - 1:
                    if new_b == 0:
                        tight[new_b] = (tight[new_b] + cnt) % MOD
                else:
                    if new_b >= 1 and new_b <= max_balance:
                        tight[new_b] = (tight[new_b] + cnt) % MOD
        
        # Process not_tight states
        for b in range(max_balance + 1):
            cnt = current_not_tight[b]
            if cnt == 0:
                continue
            max_v = N - 1
            if i < N - 1:
                min_v = 1 if b == 1 else 0
            else:
                min_v = 0
            
            # Valid v range is [min_v, max_v]
            if min_v > max_v:
                continue
            
            new_b_min = b - 1
            new_b_max = b - 1 + max_v
            if new_b_min < 0:
                new_b_min = 0
            if new_b_max > max_balance:
                new_b_max = max_balance
            if new_b_min <= new_b_max:
                not_tight[new_b_min] = (not_tight[new_b_min] + cnt) % MOD
                if new_b_max + 1 <= max_balance:
                    not_tight[new_b_max + 1] = (not_tight[new_b_max + 1] - cnt) % MOD
        
        # Compute prefix sums for tight and not_tight
        for k in range(1, max_balance + 1):
            tight[k] = (tight[k] + tight[k-1]) % MOD
            not_tight[k] = (not_tight[k] + not_tight[k-1]) % MOD
        
        # Update current states
        for k in range(max_balance + 1):
            current_tight[k] = tight[k]
            current_not_tight[k] = not_tight[k]
    
    # Sum all valid states with balance 0
    total = (current_tight[0] + current_not_tight[0]) % MOD
    print(total)

if __name__ == '__main__':
    main()