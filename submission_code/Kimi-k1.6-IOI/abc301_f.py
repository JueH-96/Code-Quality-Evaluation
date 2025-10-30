from collections import Counter

MOD = 998244353

def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    current_dp = [0] * 55
    current_dp[0] = 1  # Initial state: state 0
    
    for c in S:
        allowed_upper = []
        allowed_lower = []
        if c == '?':
            allowed_upper = [chr(ord('A') + i) for i in range(26)]
            allowed_lower = [chr(ord('a') + i) for i in range(26)]
        else:
            if c.isupper():
                allowed_upper = [c]
            else:
                allowed_lower = [c]
        
        freq_upper = Counter(allowed_upper)
        freq_lower = Counter(allowed_lower)
        
        new_dp = [0] * 55
        for s in range(55):
            if current_dp[s] == 0:
                continue
            
            # Process uppercase characters
            for char, cnt in freq_upper.items():
                if s == 0:
                    # Transition from state 0 to state1_char
                    new_s = 1 + (ord(char) - ord('A'))
                elif 1 <= s <= 26:
                    x = chr(ord('A') + (s - 1))
                    if char == x:
                        new_s = 27 + (s - 1)
                    else:
                        new_s = 1 + (ord(char) - ord('A'))
                elif 27 <= s <= 52:
                    # state2_x: uppercase stays in state2_x, others transition
                    new_s = s
                elif s == 53:
                    # state3: uppercase transitions to state4
                    new_s = 54
                else:  # s == 54
                    new_s = 54
                new_dp[new_s] = (new_dp[new_s] + current_dp[s] * cnt) % MOD
            
            # Process lowercase characters
            for char, cnt in freq_lower.items():
                if s == 0:
                    new_s = 0
                elif 1 <= s <= 26:
                    new_s = s
                elif 27 <= s <= 52:
                    new_s = 53
                elif s == 53:
                    new_s = 53
                else:  # s == 54
                    new_s = 54
                new_dp[new_s] = (new_dp[new_s] + current_dp[s] * cnt) % MOD
        
        current_dp = new_dp
    
    total = sum(current_dp[:54]) % MOD
    print(total)

if __name__ == "__main__":
    main()