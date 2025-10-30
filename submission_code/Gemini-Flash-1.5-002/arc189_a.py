MOD = 998244353

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    initial_state = [(i % 2) for i in range(1, n + 1)]
    
    if initial_state == a:
        print(1)
        return

    count = 0
    q = [(initial_state, [])]
    visited = {tuple(initial_state)}

    while q:
        curr_state, operations = q.pop(0)
        
        if curr_state == a:
            count = (count + 1) % MOD
            continue

        for l in range(n):
            for r in range(l + 2, n):
                if curr_state[l] == curr_state[r] and all(curr_state[i] != curr_state[l] for i in range(l + 1, r)):
                    next_state = curr_state[:]
                    for i in range(l + 1, r):
                        next_state[i] = curr_state[l]
                    
                    next_state_tuple = tuple(next_state)
                    if next_state_tuple not in visited:
                        visited.add(next_state_tuple)
                        q.append((next_state, operations + [(l,r)]))

    print(count)

solve()