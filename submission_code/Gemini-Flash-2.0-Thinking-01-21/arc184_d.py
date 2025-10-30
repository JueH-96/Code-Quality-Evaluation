import collections

def solve():
    n = int(input())
    balls_coords = []
    for _ in range(n):
        x, y = map(int, input().split())
        balls_coords.append({'x': x, 'y': y, 'id': _ + 1})
    
    initial_set = frozenset(range(1, n + 1))
    reachable_sets = {initial_set}
    queue = collections.deque([initial_set])
    
    while queue:
        current_set_frozenset = queue.popleft()
        current_set = set(current_set_frozenset)
        for ball_k_id in current_set:
            ball_k_info = None
            for ball in balls_coords:
                if ball['id'] == ball_k_id:
                    ball_k_info = ball
                    break
            
            next_set = set([ball_k_id])
            for ball_i_id in current_set:
                if ball_i_id == ball_k_id:
                    continue
                ball_i_info = None
                for ball in balls_coords:
                    if ball['id'] == ball_i_id:
                        ball_i_info = ball
                        break
                        
                remove_condition = (ball_i_info['x'] < ball_k_info['x'] and ball_i_info['y'] < ball_k_info['y']) or \
                                   (ball_i_info['x'] > ball_k_info['x'] and ball_i_info['y'] > ball_k_info['y'])
                if not remove_condition:
                    next_set.add(ball_i_id)
                    
            next_frozenset = frozenset(next_set)
            if next_frozenset not in reachable_sets:
                reachable_sets.add(next_frozenset)
                queue.append(next_frozenset)
                
    print(len(reachable_sets) % 998244353)

if __name__ == '__main__':
    solve()