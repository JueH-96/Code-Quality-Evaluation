def solve():
    s_segment_str = input()
    t_segment_str = input()
    s1, s2 = s_segment_str[0], s_segment_str[1]
    t1, t2 = t_segment_str[0], t_segment_str[1]
    
    side_segments = {"AB", "BC", "CD", "DE", "AE"}
    
    s_sorted_chars = sorted([s1, s2])
    s_segment_ordered_str = "".join(s_sorted_chars)
    
    t_sorted_chars = sorted([t1, t2])
    t_segment_ordered_str = "".join(t_sorted_chars)
    
    s_is_side = s_segment_ordered_str in side_segments
    t_is_side = t_segment_ordered_str in side_segments
    
    if s_is_side == t_is_side:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()