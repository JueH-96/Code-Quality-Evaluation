def solve():
    N, X, Y = map(int, input().split())
    dishes = []
    for _ in range(N):
        a, b = map(int, input().split())
        dishes.append((a, b))
    
    max_dishes = 0
    
    # Try all possible subsets
    for mask in range(1 << N):
        selected = []
        for i in range(N):
            if mask & (1 << i):
                selected.append(dishes[i])
        
        if len(selected) <= max_dishes:
            continue
            
        # Check if this subset can be arranged validly
        def can_arrange(dishes_list):
            if not dishes_list:
                return True
            
            used = [False] * len(dishes_list)
            
            def dfs(pos, sweet_sum, salt_sum):
                if pos == len(dishes_list):
                    return True
                
                for i in range(len(dishes_list)):
                    if used[i]:
                        continue
                    
                    new_sweet = sweet_sum + dishes_list[i][0]
                    new_salt = salt_sum + dishes_list[i][1]
                    
                    if new_sweet <= X and new_salt <= Y:
                        used[i] = True
                        if dfs(pos + 1, new_sweet, new_salt):
                            return True
                        used[i] = False
                
                return False
            
            return dfs(0, 0, 0)
        
        if can_arrange(selected):
            max_dishes = len(selected)
    
    return max_dishes

print(solve())