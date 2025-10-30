def solve():
    n = int(input())
    s = input()
    t = input()

    s_list = list(s)
    t_list = list(t)

    q = [(s_list + ['.', '.'], 0)]
    visited = {tuple(s_list + ['.', '.'])}

    while q:
        curr_state, curr_ops = q.pop(0)

        if curr_state[:n] == t_list:
            print(curr_ops)
            return

        empty_indices = [i for i, x in enumerate(curr_state) if x == '.']

        for i in range(n + 1):
            if curr_state[i] != '.' and curr_state[i+1] != '.':
                next_state = curr_state[:]
                next_state[empty_indices[0]] = next_state[i]
                next_state[empty_indices[1]] = next_state[i+1]
                next_state[i] = '.'
                next_state[i+1] = '.'
                
                if tuple(next_state) not in visited:
                    visited.add(tuple(next_state))
                    q.append((next_state, curr_ops + 1))

    print(-1)

solve()