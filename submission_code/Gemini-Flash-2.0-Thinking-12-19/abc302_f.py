import collections

def solve():
    n, m = map(int, input().split())
    initial_sets = []
    for _ in range(n):
        a_line = list(map(int, input().split()))
        initial_sets.append(frozenset(a_line[1:]))
    
    start_state = tuple(sorted(initial_sets, key=lambda s: tuple(sorted(list(s)))))
    queue = collections.deque([(start_state, 0)])
    visited_states = {start_state}
    
    min_ops = -1
    
    while queue:
        current_state, operations_count = queue.popleft()
        
        found_target_set = False
        for s in current_state:
            if 1 in s and m in s:
                found_target_set = True
                break
        if found_target_set:
            min_ops = operations_count
            break
            
        next_possible_operations = []
        sets_list = list(current_state)
        num_sets = len(sets_list)
        for i in range(num_sets):
            for j in range(i + 1, num_sets):
                set1 = sets_list[i]
                set2 = sets_list[j]
                if set1.intersection(set2):
                    next_set = set1.union(set2)
                    remaining_sets = []
                    for k in range(num_sets):
                        if k != i and k != j:
                            remaining_sets.append(sets_list[k])
                    next_state_sets = tuple(sorted(remaining_sets + [next_set], key=lambda s: tuple(sorted(list(s)))))
                    if next_state_sets not in visited_states:
                        visited_states.add(next_state_sets)
                        queue.append((next_state_sets, operations_count + 1))
                        
        if min_ops != -1:
            break
            
    print(min_ops)

if __name__ == '__main__':
    solve()