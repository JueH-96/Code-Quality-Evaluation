import collections

def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))
    
    initial_state = (tuple(a), tuple(b))
    target_red_balls = [0] * n
    target_blue_balls = [0] * n
    total_red_balls = sum(a)
    total_blue_balls = sum(b)
    target_red_balls[x-1] = total_red_balls
    target_blue_balls[x-1] = total_blue_balls
    target_state = (tuple(target_red_balls), tuple(target_blue_balls))
    
    u = set(range(1, n + 1))
    u.remove(x)
    s_star = set(u)
    changed = True
    while changed:
        changed = False
        s_next = set()
        for i in s_star:
            if p[i-1] in s_star and q[i-1] in s_star:
                s_next.add(i)
        if s_next != s_star:
            changed = True
            s_star = s_next
            
    impossible = False
    if s_star:
        balls_in_s_star = 0
        for i in s_star:
            balls_in_s_star += a[i-1] + b[i-1]
        if balls_in_s_star > 0:
            impossible = True
            
    if impossible:
        print("-1")
    else:
        if initial_state == target_state:
            print(0)
            return
            
        queue = collections.deque([(initial_state, 0)])
        visited_states = {initial_state}
        
        while queue:
            current_state, operations_count = queue.popleft()
            if current_state == target_state:
                print(operations_count)
                return
                
            for i in range(1, n + 1):
                current_red_balls = list(current_state[0])
                current_blue_balls = list(current_state[1])
                red_balls_from_i = current_red_balls[i-1]
                blue_balls_from_i = current_blue_balls[i-1]
                current_red_balls[i-1] = 0
                current_blue_balls[i-1] = 0
                current_red_balls[p[i-1]-1] += red_balls_from_i
                current_blue_balls[q[i-1]-1] += blue_balls_from_i
                next_state = (tuple(current_red_balls), tuple(current_blue_balls))
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    queue.append((next_state, operations_count + 1))
                    
        print("-1") # Should not reach here if not impossible

if __name__ == '__main__':
    solve()