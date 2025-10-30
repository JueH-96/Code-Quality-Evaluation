import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    dp_prev = defaultdict(int)
    dp_prev[""] = 1  # Initial state is empty string

    for i in range(N):
        dp_next = defaultdict(int)
        current_char = S[i]
        possible_chars = ['A', 'B'] if current_char == '?' else [current_char]

        for state in dp_prev:
            count = dp_prev[state]
            for c in possible_chars:
                # Build new_state
                if len(state) < K-1:
                    new_state = state + c
                else:
                    new_state = state[1:] + c
                
                # Check palindrome window if applicable
                window_ok = True
                if i >= K-1:  # 0-based index, check window of size K
                    window = state + c  # state is K-1 characters, c is the new char
                    if window == window[::-1]:
                        window_ok = False
                
                if not window_ok:
                    continue
                
                dp_next[new_state] = (dp_next[new_state] + count) % MOD
        
        dp_prev = dp_next
        if not dp_prev:
            break  # No need to proceed further

    print(sum(dp_prev.values()) % MOD)

if __name__ == "__main__":
    main()