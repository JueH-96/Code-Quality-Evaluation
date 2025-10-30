import collections

def solve():
    n, m = map(int, input().split())
    initial_sets = []
    for _ in range(n):
        line = list(map(int, input().split()))
        initial_sets.append(frozenset(line[1:]))
    
    if any(1 in s and m in s for s in initial_sets):
        print(0)
        return
        
    start_state = tuple(initial_sets)
    queue = collections.deque([(start_state, 0)])
    visited_states = {start_state}
    
    min_operations = float('inf')
    found_solution = False
    
    while queue:
        current_state, operations_count = queue.popleft()
        
        for s in current_state:
            if 1 in s and m in s:
                min_operations = min(min_operations, operations_count)
                found_solution = True
                
        if operations_count >= n - 1:
            continue
            
        for i in range(len(current_state)):
            for j in range(i + 1, len(current_state)):
                set1 = current_state[i]
                set2 = current_state[j]
                if set1.intersection(set2):
                    next_set = set1.union(set2)
                    next_state_list = list(current_state)
                    next_state_list.pop(j)
                    next_state_list.pop(i)
                    next_state_list.append(next_set)
                    next_state_tuple = tuple(sorted(next_state_list, key=lambda x: tuple(sorted(list(x)))))
                    if next_state_tuple not in visited_states:
                        visited_states.add(next_state_tuple)
                        queue.append((next_state_tuple, operations_count + 1))
                        
    if found_solution:
        print(min_operations)
    else:
        print(-1)

if __name__ == '__main__':
    solve()