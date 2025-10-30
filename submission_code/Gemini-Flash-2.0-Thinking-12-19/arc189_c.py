import collections

def solve():
    n, x = map(int, input().split())
    initial_a = list(map(int, input().split()))
    initial_b = list(map(int, input().split()))
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))
    
    u = set(range(1, n + 1))
    u.remove(x)
    s_inf = set(u)
    
    while True:
        s_next = set()
        for i in s_inf:
            if p[i-1] in s_inf and q[i-1] in s_inf:
                s_next.add(i)
        if s_next == s_inf:
            break
        s_inf = s_next
        
    initial_balls_in_s_inf = 0
    for i in s_inf:
        initial_balls_in_s_inf += initial_a[i-1] + initial_b[i-1]
        
    if s_inf and initial_balls_in_s_inf > 0:
        print("-1")
        return
        
    initial_state = tuple(initial_a + initial_b)
    target_state_condition = lambda state: all(state[i-1] == 0 and state[n+i-1] == 0 for i in range(1, n + 1) if i != x)
    
    if target_state_condition(initial_state):
        print(0)
        return
        
    queue = collections.deque([(initial_state, 0)])
    visited_states = {initial_state}
    
    while queue:
        current_state, operations_count = queue.popleft()
        
        if target_state_condition(current_state):
            print(operations_count)
            return
            
        current_a = list(current_state[:n])
        current_b = list(current_state[n:])
        
        for i in range(1, n + 1):
            next_a = list(current_a)
            next_b = list(current_b)
            
            red_balls = next_a[i-1]
            blue_balls = next_b[i-1]
            
            next_a[i-1] = 0
            next_b[i-1] = 0
            
            next_a[p[i-1]-1] += red_balls
            next_b[q[i-1]-1] += blue_balls
            
            next_state = tuple(next_a + next_b)
            if next_state not in visited_states:
                visited_states.add(next_state)
                queue.append((next_state, operations_count + 1))
                
    print("-1") # Should not reach here based on the problem description if not impossible from condition check

if __name__ == '__main__':
    solve()