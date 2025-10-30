def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    reach_sets = []
    for i_index in range(1, n + 1):
        reach_i = set()
        visited = set()
        queue = [i_index]
        visited.add(i_index)
        reach_i.add(i_index)
        
        while queue:
            u_index = queue.pop(0)
            for v_index in range(1, n + 1):
                if abs(u_index - v_index) <= k:
                    if v_index not in visited:
                        visited.add(v_index)
                        reach_i.add(v_index)
                        queue.append(v_index)
        reach_sets.append(reach_i)
        
    possible_values_list = []
    for i_index in range(1, n + 1):
        possible_values = set()
        reach_i = reach_sets[i_index-1]
        for j_index in reach_i:
            possible_values.add(a[j_index-1])
        possible_values_list.append(possible_values)
        
    possible_to_make_b = True
    for i_index in range(1, n + 1):
        if b[i_index-1] not in possible_values_list[i_index-1]:
            possible_to_make_b = False
            break
            
    if possible_to_make_b:
        print("Yes")
    else:
        print("No")

t = int(input())
for _ in range(t):
    solve()