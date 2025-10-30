import sys
from collections import defaultdict

MOD = 998244353

def main():
    S = sys.stdin.readline().strip()
    current_dp = {0: 1}  # initial state: state 0 with count 1

    for c in S:
        next_dp = defaultdict(int)
        if c == '?':
            possible_chars = [chr(ord('A') + i) for i in range(26)] + [chr(ord('a') + i) for i in range(26)]
        else:
            possible_chars = [c]
        
        for state in current_dp:
            count = current_dp[state]
            for char in possible_chars:
                new_state = None
                if state == 0:
                    if char.isupper():
                        new_state = (1, char)
                    else:
                        new_state = 0
                elif isinstance(state, tuple) and state[0] == 1:
                    x = state[1]
                    if char.isupper():
                        if char == x:
                            new_state = 2
                        else:
                            new_state = (1, char)
                    else:
                        new_state = state
                elif state == 2:
                    if char.islower():
                        new_state = 3
                    else:
                        new_state = 2
                elif state == 3:
                    if char.isupper():
                        new_state = 4
                    else:
                        new_state = 3
                elif state == 4:
                    new_state = 4
                else:
                    continue  # invalid state, skip
                
                next_dp[new_state] = (next_dp[new_state] + count) % MOD
        
        current_dp = next_dp
    
    result = 0
    for state in current_dp:
        if state != 4:
            result = (result + current_dp[state]) % MOD
    print(result)

if __name__ == '__main__':
    main()