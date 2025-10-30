from collections import deque

def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    S = data[1].strip()
    T = data[2].strip()
    
    if sorted(S) != sorted(T):
        print(-1)
        return
        
    total_cells = n + 2
    initial_state = list(S + "..")
    goal_state = list(T + "..")
    
    if initial_state == goal_state:
        print(0)
        return
        
    visited = {}
    queue = deque()
    start = "".join(initial_state)
    visited[start] = 0
    queue.append((start, 0))
    
    while queue:
        current_str, steps = queue.popleft()
        if current_str.startswith(T):
            print(steps)
            return
            
        empty_pos = current_str.rfind('..')
        for i in range(total_cells - 1):
            if i == empty_pos or i+1 == empty_pos:
                continue
            if current_str[i] == '.' or current_str[i+1] == '.':
                continue
                
            j = i
            while j < total_cells - 1:
                if current_str[j] == '.' or current_str[j+1] == '.':
                    break
                next_stones = current_str[j:j+2]
                if next_stones[0] + next_stones[1] == "..":
                    break
                    
                new_list = list(current_str)
                new_list[empty_pos] = next_stones[0]
                new_list[empty_pos+1] = next_stones[1]
                new_list[j] = '.'
                new_list[j+1] = '.'
                new_state = "".join(new_list)
                
                if new_state not in visited:
                    visited[new_state] = steps + 1
                    queue.append((new_state, steps + 1))
                
                j += 1
                    
    print(-1)

if __name__ == "__main__":
    main()