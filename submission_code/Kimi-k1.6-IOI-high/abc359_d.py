import sys
from collections import defaultdict
import itertools

MOD = 998244353

def main():
    n, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    
    # Generate initial states for the first k-1 characters
    possible_choices = []
    for i in range(k-1):
        c = s[i] if i < len(s) else '?'
        if c == 'A':
            possible_choices.append(['A'])
        elif c == 'B':
            possible_choices.append(['B'])
        else:
            possible_choices.append(['A', 'B'])
    
    initial_states = list(itertools.product(*possible_choices))
    prev_dp = defaultdict(int)
    for state in initial_states:
        prev_dp[state] += 1
    
    # Process each position from k-1 to n-1
    for i in range(k-1, n):
        current_char = s[i] if i < len(s) else '?'
        if current_char == '?':
            possible_chars = ['A', 'B']
        else:
            possible_chars = [current_char]
        
        new_dp = defaultdict(int)
        for state in prev_dp:
            count = prev_dp[state]
            for c in possible_chars:
                candidate = state + (c,)
                # Check if candidate is a palindrome
                is_palin = True
                for j in range(k // 2):
                    if candidate[j] != candidate[k-1 - j]:
                        is_palin = False
                        break
                if is_palin:
                    continue
                # Valid transition, update new state
                new_state = state[1:] + (c,)
                new_dp[new_state] = (new_dp[new_state] + count) % MOD
        prev_dp = new_dp
    
    total = sum(prev_dp.values()) % MOD
    print(total)

if __name__ == "__main__":
    main()