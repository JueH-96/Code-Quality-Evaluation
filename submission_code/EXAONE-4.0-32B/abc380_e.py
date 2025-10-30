import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    
    parent = list(range(n+1))
    dsu_left = list(range(n+1))
    dsu_right = list(range(n+1))
    dsu_color = list(range(n+1))
    count = [0] * (n+1)
    for i in range(1, n+1):
        count[i] = 1
        
    def find(x):
        root = x
        while root != parent[root]:
            root = parent[root]
        while x != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    out_lines = []
    for _ in range(q):
        t = next(it)
        if t == '1':
            x = int(next(it))
            c = int(next(it))
            rep_x = find(x)
            L = dsu_left[rep_x]
            R = dsu_right[rep_x]
            old_color = dsu_color[rep_x]
            if old_color == c:
                continue
            count[old_color] -= (R - L + 1)
            
            if L > 1:
                rep_left = find(L-1)
                if dsu_color[rep_left] == c:
                    count[c] -= (dsu_right[rep_left] - dsu_left[rep_left] + 1)
                    parent[rep_left] = rep_x
                    dsu_left[rep_x] = dsu_left[rep_left]
                    
            if R < n:
                rep_right = find(R+1)
                if dsu_color[rep_right] == c:
                    count[c] -= (dsu_right[rep_right] - dsu_left[rep_right] + 1)
                    parent[rep_right] = rep_x
                    dsu_right[rep_x] = dsu_right[rep_right]
                    
            dsu_color[rep_x] = c
            count[c] += (dsu_right[rep_x] - dsu_left[rep_x] + 1)
            
        else:
            c = int(next(it))
            out_lines.append(str(count[c]))
            
    print("
".join(out_lines))

if __name__ == '__main__':
    main()