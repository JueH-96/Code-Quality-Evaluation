from collections import deque

def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))
    
    x -= 1
    
    initial_state = tuple((a[i], b[i]) for i in range(n))
    
    if all(sum(initial_state[i]) == 0 for i in range(n) if i != x):
        print(0)
        return

    q = deque([(initial_state, 0)])
    visited = {initial_state}

    while q:
        current_state, steps = q.popleft()

        all_empty_except_x = True
        for i in range(n):
            if i != x and sum(current_state[i]) != 0:
                all_empty_except_x = False
                break
        if all_empty_except_x:
            print(steps)
            return
        
        for i in range(n):
            next_state_list = list(list(pair) for pair in current_state)
            
            red_balls = next_state_list[i][0]
            blue_balls = next_state_list[i][1]
            
            next_state_list[i][0] = 0
            next_state_list[i][1] = 0
            
            next_state_list[p[i]-1][0] += red_balls
            next_state_list[q[i]-1][1] += blue_balls
            
            next_state = tuple(tuple(pair) for pair in next_state_list)
            
            if next_state not in visited:
                visited.add(next_state)
                q.append((next_state, steps + 1))
    
    print(-1)

solve()