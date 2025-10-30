MOD = 998244353

def main():
    import sys
    S = sys.stdin.read().strip()
    N = len(S)
    
    # Initialize states
    state0 = 1
    state1 = 0  # sum over all A
    state2 = 0  # sum over all A
    state3 = 0  # sum over all A and a
    
    for i in range(N):
        c = S[i]
        new_state0 = 0
        new_state1 = 0
        new_state2 = 0
        new_state3 = 0
        
        # From state0
        if c == '?':
            choices_any = 52
            choices_upper = 26
        elif c.isupper():
            choices_any = 1
            choices_upper = 1
        else:
            choices_any = 1
            choices_upper = 0
        new_state0 = (new_state0 + state0 * choices_any) % MOD
        if c == '?':
            new_state1 = (new_state1 + state0 * 26) % MOD
        elif c.isupper():
            new_state1 = (new_state1 + state0 * 1) % MOD
        
        # From state1
        if c == '?':
            choices_not_A = 51  # 52 total - 1 specific A
            choices_A = 1
        elif c.isupper():
            choices_not_A = 1 if c != S[i] else 0
            choices_A = 1 if c == S[i] else 0
        else:
            choices_not_A = 1
            choices_A = 0
        new_state1 = (new_state1 + state1 * choices_not_A) % MOD
        if c == '?':
            new_state2 = (new_state2 + state1 * 1) % MOD
        elif c.isupper():
            new_state2 = (new_state2 + state1 * 1) % MOD
        
        # From state2
        if c == '?':
            choices_not_a = 26  # uppercase letters
            choices_a = 26  # lowercase letters
        elif c.islower():
            choices_not_a = 1 if c != 'a' else 0
            choices_a = 1 if c == 'a' else 0
        else:
            choices_not_a = 1
            choices_a = 0
        new_state2 = (new_state2 + state2 * choices_not_a) % MOD
        if c == '?':
            new_state3 = (new_state3 + state2 * 26) % MOD
        elif c.islower():
            new_state3 = (new_state3 + state2 * 1) % MOD
        
        # From state3
        if c == '?':
            choices_not_B = 51  # 52 total - 1 specific B
        elif c.isupper():
            choices_not_B = 1 if c != 'B' else 0
        else:
            choices_not_B = 1
        new_state3 = (new_state3 + state3 * choices_not_B) % MOD
        
        # Update states
        state0, state1, state2, state3 = new_state0, new_state1, new_state2, new_state3
    
    # Total valid strings
    total = (state0 + state1 + state2 + state3) % MOD
    print(total)

if __name__ == "__main__":
    main()