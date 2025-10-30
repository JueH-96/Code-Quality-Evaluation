def solve():
    n = int(input())
    balls = []
    for _ in range(n):
        x, y = map(int, input().split())
        balls.append((x, y))

    def remove_balls(current_balls, k):
        new_balls = []
        for i in range(len(current_balls)):
            if i == k:
                new_balls.append(current_balls[i])
                continue
            x_i, y_i = current_balls[i]
            x_k, y_k = current_balls[k]
            if not ((x_i < x_k and y_i < y_k) or (x_i > x_k and y_i > y_k)):
                new_balls.append(current_balls[i])
        return new_balls

    def get_all_possible_sets(current_balls):
        if not current_balls:
            return {tuple()}
        
        all_sets = set()
        for i in range(len(current_balls)):
            new_balls = remove_balls(current_balls, i)
            
            if tuple(sorted(new_balls)) not in all_sets:
                all_sets.add(tuple(sorted(new_balls)))
                
                if len(new_balls) > 0:
                    all_sets.update(get_all_possible_sets(new_balls))
        
        return all_sets

    all_possible_sets = get_all_possible_sets(balls)
    print(len(all_possible_sets) % 998244353)

solve()