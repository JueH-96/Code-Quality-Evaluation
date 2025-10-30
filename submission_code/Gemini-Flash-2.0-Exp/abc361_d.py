def solve():
    n = int(input())
    s = input()
    t = input()

    if s.count('B') != t.count('B') or s.count('W') != t.count('W'):
        print(-1)
        return

    def get_moves(start, target, n):
        q = [(start, 0)]
        visited = {start}
        
        while q:
            curr_state, moves = q.pop(0)
            
            if curr_state == target:
                return moves
            
            for i in range(n + 1):
                if curr_state[i] != '.' and curr_state[i+1] != '.':
                    
                    empty_indices = []
                    for j in range(n + 2):
                        if curr_state[j] == '.':
                            empty_indices.append(j)
                    
                    if len(empty_indices) >= 2:
                        k1, k2 = empty_indices[0], empty_indices[1]
                        
                        new_state_list = list(curr_state)
                        
                        new_state_list[k1] = curr_state[i]
                        new_state_list[k2] = curr_state[i+1]
                        new_state_list[i] = '.'
                        new_state_list[i+1] = '.'
                        
                        new_state = "".join(new_state_list)
                        
                        if new_state not in visited:
                            q.append((new_state, moves + 1))
                            visited.add(new_state)
                            
        return float('inf')

    initial_state = s + ".."
    target_state = t + ".."

    result = get_moves(initial_state, target_state, n)

    if result == float('inf'):
        print(-1)
    else:
        print(result)

solve()