def solve():
    n = int(input())
    a = input()
    b = input()

    a_list = [int(c) for c in a]
    b_list = [int(c) for c in b]

    def check(config, target):
        for i in range(n):
            if target[i] == 1 and config[i] == 0:
                return False
            if target[i] == 0 and config[i] > 0:
                return False
        return True

    def apply_operation(config, i):
        new_config = list(config)
        for j in range(n):
            if config[j] > 0:
                count = config[j]
                new_config[j] -= config[j]
                
                if i < j:
                    new_pos = j - 1
                elif i > j:
                    new_pos = j + 1
                else:
                    new_pos = j
                
                if 0 <= new_pos < n:
                    new_config[new_pos] += count
        return new_config

    q = [(a_list, 0)]
    visited = {tuple(a_list)}
    
    while q:
        curr_config, curr_moves = q.pop(0)
        
        if check(curr_config, b_list):
            print(curr_moves)
            return

        for i in range(1, n + 1):
            next_config = apply_operation(curr_config, i - 1)
            
            if tuple(next_config) not in visited:
                visited.add(tuple(next_config))
                q.append((next_config, curr_moves + 1))

    print(-1)

t = int(input())
for _ in range(t):
    solve()