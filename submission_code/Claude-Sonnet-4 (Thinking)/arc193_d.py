from collections import deque

def solve(n, a, b):
    count_a = a.count('1')
    count_b = b.count('1')
    
    if count_a < count_b:
        return -1
    
    # Initial configuration
    init_config = [0] * n
    for i in range(n):
        if a[i] == '1':
            init_config[i] = 1
    init_config = tuple(init_config)
    
    # Check if current configuration is valid
    def is_valid(config):
        for i in range(n):
            if b[i] == '1' and config[i] == 0:
                return False
            if b[i] == '0' and config[i] > 0:
                return False
        return True
    
    if is_valid(init_config):
        return 0
    
    # BFS to find minimum operations
    queue = deque([(init_config, 0)])
    visited = set([init_config])
    
    while queue:
        config, ops = queue.popleft()
        
        # Try each possible operation
        for i in range(n):
            new_config = [0] * n
            for j in range(n):
                if config[j] > 0:
                    if j < i:
                        new_config[j + 1] += config[j]
                    elif j > i:
                        new_config[j - 1] += config[j]
                    else:
                        new_config[j] += config[j]
            
            new_config_tuple = tuple(new_config)
            
            if is_valid(new_config_tuple):
                return ops + 1
            
            if new_config_tuple not in visited:
                visited.add(new_config_tuple)
                queue.append((new_config_tuple, ops + 1))
    
    return -1

t = int(input())
for _ in range(t):
    n = int(input())
    a = input().strip()
    b = input().strip()
    print(solve(n, a, b))