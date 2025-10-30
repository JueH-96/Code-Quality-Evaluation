def solve():
    n = int(input())
    s = input()
    t = input()

    if s.count('B') != t.count('B') or s.count('W') != t.count('W'):
        print(-1)
        return

    q = [(list(s) + ['.', '.'], 0)]
    visited = {tuple(list(s) + ['.', '.'])}

    while q:
        curr_state, steps = q.pop(0)
        
        target_state = list(t) + ['.', '.']
        if curr_state == target_state:
            print(steps)
            return

        for i in range(n + 1):
            if curr_state[i] != '.' and curr_state[i+1] != '.':
                
                empty_indices = [idx for idx, val in enumerate(curr_state) if val == '.']
                if len(empty_indices) >= 2:
                    
                    new_state = curr_state[:]
                    new_state[empty_indices[0]] = curr_state[i]
                    new_state[empty_indices[1]] = curr_state[i+1]
                    new_state[i] = '.'
                    new_state[i+1] = '.'
                    
                    if tuple(new_state) not in visited:
                        visited.add(tuple(new_state))
                        q.append((new_state, steps + 1))
    
    print(-1)

solve()