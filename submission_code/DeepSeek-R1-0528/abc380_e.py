import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); q = int(next(it))
    
    cnt = [0] * (n + 1)
    for i in range(1, n + 1):
        cnt[i] = 1
        
    right_rep = list(range(n + 1))
    left_from_right = list(range(n + 1))
    color_by_right = list(range(n + 1))
    
    def find_right(x):
        root = x
        while root != right_rep[root]:
            root = right_rep[root]
        current = x
        while current != root:
            nxt = right_rep[current]
            right_rep[current] = root
            current = nxt
        return root

    output_lines = []
    
    for _ in range(q):
        t = next(it)
        if t == '1':
            x = int(next(it)); c = int(next(it))
            r0 = find_right(x)
            l0 = left_from_right[r0]
            old_color = color_by_right[r0]
            size = r0 - l0 + 1
            
            if old_color != c:
                cnt[old_color] -= size
                cnt[c] += size
                color_by_right[r0] = c
                
            if l0 > 1:
                l_prev = l0 - 1
                r_prev = find_right(l_prev)
                l_prev_seg = left_from_right[r_prev]
                color_prev = color_by_right[r_prev]
                
                if color_prev == c:
                    right_rep[r_prev] = r0
                    left_from_right[r0] = l_prev_seg
                    
            if r0 < n:
                r_next = r0 + 1
                r_next_seg = find_right(r_next)
                l_next_seg = left_from_right[r_next_seg]
                color_next = color_by_right[r_next_seg]
                
                if color_next == c:
                    current_left = left_from_right[r0]
                    right_rep[r0] = r_next_seg
                    left_from_right[r_next_seg] = current_left
                    
        else:
            c = int(next(it))
            output_lines.append(str(cnt[c]))
            
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()