def solve():
    s = input()
    n = len(s)
    num_states = 79 + 1
    dp = [[0] * num_states for _ in range(n + 1)]
    dp[0][0] = 1
    mod = 998244353
    
    def get_next_state(current_state, char):
        if current_state == 79:
            return 79
        if 1 <= current_state <= 26: # state 1-U (U is (current_state)-th uppercase letter)
            u = chr(ord('A') + current_state - 1)
            if 'A' <= char <= 'Z':
                v = char
                if v == u:
                    return 26 + current_state # state 2-U 
                else:
                    return ord(v) - ord('A') + 1 # state 1-V 
            elif 'a' <= char <= 'z':
                return current_state # stay in state 1-U
            else:
                return current_state
        elif 27 <= current_state <= 52: # state 2-U (U is (current_state-26)-th uppercase letter)
            # u = chr(ord('A') + (current_state - 27))
            if 'A' <= char <= 'Z':
                return current_state # stay in state 2-U 
            elif 'a' <= char <= 'z':
                return 52 + (current_state - 26) # state 3-U
            else:
                return current_state
        elif 53 <= current_state <= 78: # state 3-U (U is (current_state-52)-th uppercase letter)
            # u = chr(ord('A') + (current_state - 53))
            if 'A' <= char <= 'Z':
                return 79 # state 4 (DDoS)
            elif 'a' <= char <= 'z':
                return current_state # stay in state 3-U
            else:
                return current_state
        elif current_state == 0: # state 0
            if 'A' <= char <= 'Z':
                return ord(char) - ord('A') + 1 # state 1-U 
            elif 'a' <= char <= 'z':
                return 0 # stay in state 0
            else:
                return 0
        else: # state 79, or invalid state. 
            return -1

    for i in range(n):
        for state in range(num_states):
            if dp[i][state] > 0:
                char_at_i = s[i]
                if char_at_i == '?':
                    for upper_char_code in range(ord('A'), ord('Z') + 1):
                        next_char = chr(upper_char_code)
                        next_state = get_next_state(state, next_char)
                        dp[i+1][next_state] = (dp[i+1][next_state] + dp[i][state]) % mod
                    for lower_char_code in range(ord('a'), ord('z') + 1):
                        next_char = chr(lower_char_code)
                        next_state = get_next_state(state, next_char)
                        dp[i+1][next_state] = (dp[i+1][next_state] + dp[i][state]) % mod
                else:
                    next_state = get_next_state(state, char_at_i)
                    dp[i+1][next_state] = (dp[i+1][next_state] + dp[i][state]) % mod

    ans = 0
    for state in range(num_states - 1):
        ans = (ans + dp[n][state]) % mod
    print(ans)

if __name__ == '__main__':
    solve()