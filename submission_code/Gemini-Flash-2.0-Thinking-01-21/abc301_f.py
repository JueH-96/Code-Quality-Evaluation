def solve():
    s = input()
    n = len(s)
    states = {}
    initial_state = ('#', '#', '#')
    states[initial_state] = 1
    
    def is_uppercase_char(char):
        return 'A' <= char <= 'Z'
    
    def is_lowercase_char(char):
        return 'a' <= char <= 'z'
        
    def is_ddos_ending(c1, c2, c3, c4):
        if c1 == '#' or c2 == '#' or c3 == '#' or c4 == '#':
            return False
        return is_uppercase_char(c1) and is_uppercase_char(c2) and c1 == c2 and is_lowercase_char(c3) and is_uppercase_char(c4)
        
    for i in range(n):
        next_states = {}
        possible_chars = []
        if s[i] == '?':
            for char_code in range(ord('A'), ord('Z') + 1):
                possible_chars.append(chr(char_code))
            for char_code in range(ord('a'), ord('z') + 1):
                possible_chars.append(chr(char_code))
        else:
            possible_chars = [s[i]]
            
        for current_state, count in states.items():
            c1, c2, c3 = current_state
            for next_char in possible_chars:
                next_state_tuple = (c2, c3, next_char)
                if not is_ddos_ending(c1, c2, c3, next_char):
                    next_states[next_state_tuple] = next_states.get(next_state_tuple, 0) + count
                    
        states = {}
        for state, count in next_states.items():
            states[state] = count % 998244353
            
    total_count = 0
    for count in states.values():
        total_count = (total_count + count) % 998244353
        
    print(total_count)

if __name__ == '__main__':
    solve()