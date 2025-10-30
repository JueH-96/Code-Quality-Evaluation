import sys

def solve():
    N = int(sys.stdin.readline())
    # A_list[i] stores A_{i+1} from problem statement (1-indexed A_y)
    A_list = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # N_inv = N^(-1) mod MOD using Fermat's Little Theorem (MOD is prime)
    N_inv = pow(N, MOD - 2, MOD)

    # E_val[k] stores the expected future salary if current x = k.
    # E_val has size N+1 to store E_0, ..., E_N.
    # E_val[N] (expected salary if current x = N) is 0, as process terminates.
    E_val = [0] * (N + 1) 
    # E_val[N] is already 0 by initialization.

    # current_S_val stores sum_{y=x+1 to N} (A_y_problem + E_y_state)
    # This is the numerator S_x in E_x = (1/N) * S_x.
    # It's calculated iteratively by adding terms from right to left.
    current_S_val = 0 
    
    # Iterate i_state from N-1 down to 0.
    # i_state is the current value of 'x' for which we are calculating E_val[x].
    for i_state in range(N - 1, -1, -1):
        # The term to add to the sum is (A_{i_state+1}_problem + E_{i_state+1}_state).
        # A_{i_state+1}_problem corresponds to A_list[i_state].
        # E_{i_state+1}_state corresponds to E_val[i_state+1].
        
        term_to_add = (A_list[i_state] + E_val[i_state + 1])
        term_to_add %= MOD 
        
        # Update current_S_val:
        # current_S_val (from previous iteration for i_state+1) was sum_{y=i_state+2 to N} (...)
        # Add term_to_add to get sum_{y=i_state+1 to N} (...)
        current_S_val = (current_S_val + term_to_add)
        current_S_val %= MOD
            
        # Calculate E_val[i_state] = (current_S_val * N_inv)
        E_val[i_state] = (current_S_val * N_inv)
        E_val[i_state] %= MOD

    # The result is E_val[0], the expected salary starting from state x=0.
    print(E_val[0])

if __name__ == '__main__':
    solve()