import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    
    initial_colors_str = sys.stdin.readline().split()
    
    boxes_content = [set() for _ in range(N)]
    for i in range(N):
        boxes_content[i] = {int(initial_colors_str[i])}

    output_lines = []

    for _ in range(Q):
        a_str, b_str = sys.stdin.readline().split()
        u = int(a_str) - 1  # 0-indexed source box
        v = int(b_str) - 1  # 0-indexed destination box

        set_u = boxes_content[u]
        set_v = boxes_content[v]
        
        # If set_u is empty, no balls to move. Result is just current size of set_v.
        # This check is implicitly handled by the logic below as len(empty_set) is 0.
        # if not set_u:
        #    output_lines.append(str(len(set_v)))
        #    continue
            
        if len(set_u) <= len(set_v):
            # set_u is smaller or equal. Iterate set_u, add its elements to set_v.
            # Box v (destination) keeps its set object (set_v), which is modified.
            set_v.update(set_u)
            # Box u (source) becomes empty (gets a new empty set object).
            boxes_content[u] = set()
            # boxes_content[v] already points to set_v, which has been updated.
        else:
            # set_u is larger. Iterate set_v, add its elements to set_u.
            # Box u's set object (set_u) is modified.
            set_u.update(set_v)
            # Box v (destination) now takes ownership of set_u's object (the merged set).
            boxes_content[v] = set_u
            # Box u (source) becomes empty (gets a new empty set object).
            boxes_content[u] = set()
        
        output_lines.append(str(len(boxes_content[v])))

    sys.stdout.write("
".join(output_lines) + "
")

if __name__ == '__main__':
    main()