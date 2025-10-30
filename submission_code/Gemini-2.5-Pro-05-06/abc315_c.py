import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline()) 
    
    flavor_to_S_list = defaultdict(list)
    for _ in range(N):
        f, s = map(int, sys.stdin.readline().split())
        flavor_to_S_list[f].append(s)
        
    max_val_same_flavor = 0
    list_of_max_S_for_flavors = [] 
    
    for flavor_id in flavor_to_S_list:
        s_list = flavor_to_S_list[flavor_id]
        s_list.sort(reverse=True) 
        
        if s_list: # This check is technically redundant given how defaultdict is used
            list_of_max_S_for_flavors.append(s_list[0])
            
        if len(s_list) >= 2:
            current_sat = s_list[0] + s_list[1] // 2
            if current_sat > max_val_same_flavor:
                max_val_same_flavor = current_sat
                
    max_val_diff_flavor = 0
    if len(list_of_max_S_for_flavors) >= 2:
        list_of_max_S_for_flavors.sort(reverse=True)
        max_val_diff_flavor = list_of_max_S_for_flavors[0] + list_of_max_S_for_flavors[1]
        
    print(max(max_val_same_flavor, max_val_diff_flavor))

solve()