def can_match(N, M, S, T):
    def check(pos, curr):
        if pos >= N:
            return curr == S
        
        # Try not replacing at current position
        if curr[pos] != '#':
            return check(pos + 1, curr)
            
        # Try replacing M chars starting at current position
        if pos + M <= N:
            new_str = curr[:pos] + T + curr[pos+M:]
            if new_str[:pos+1] == S[:pos+1]:
                if check(pos + 1, new_str):
                    return True
                    
        # Try skipping current position
        if curr[pos] == '#':
            return check(pos + 1, curr)
            
        return False

    # Start with string of all #
    X = '#' * N
    return check(0, X)

N, M = map(int, input().split())
S = input()
T = input()

if can_match(N, M, S, T):
    print("Yes")
else:
    print("No")