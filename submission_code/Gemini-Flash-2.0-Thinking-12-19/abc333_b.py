def solve():
    s_segment_str = input()
    t_segment_str = input()
    s1 = s_segment_str[0]
    s2 = s_segment_str[1]
    t1 = t_segment_str[0]
    t2 = t_segment_str[1]
    
    vertex_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    s1_index = vertex_indices[s1]
    s2_index = vertex_indices[s2]
    t1_index = vertex_indices[t1]
    t2_index = vertex_indices[t2]
    
    s_diff = abs(s1_index - s2_index)
    t_diff = abs(t1_index - t2_index)
    
    s_type = ""
    if s_diff == 1 or s_diff == 4:
        s_type = "side"
    elif s_diff == 2 or s_diff == 3:
        s_type = "diagonal"
        
    t_type = ""
    if t_diff == 1 or t_diff == 4:
        t_type = "side"
    elif t_diff == 2 or t_diff == 3:
        t_type = "diagonal"
        
    if s_type == t_type:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()