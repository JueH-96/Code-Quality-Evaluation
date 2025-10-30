import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    if S == T:
        print(0)
        return
    
    s_map = {}
    for i in range(N):
        s = S[i]
        t = T[i]
        if s in s_map:
            if s_map[s] != t:
                print(-1)
                return
        else:
            s_map[s] = t
    
    # Remove self mappings
    new_s_map = {k: v for k, v in s_map.items() if k != v}
    if not new_s_map:
        print(0)
        return
    
    total_edges = len(new_s_map)
    
    # Count cycles
    cycle_count = 0
    visited = set()
    for c in list(new_s_map.keys()):
        if c in visited:
            continue
        current_path = set()
        current = c
        while True:
            if current not in new_s_map:
                break
            if current in visited:
                break
            visited.add(current)
            current_path.add(current)
            current = new_s_map[current]
            if current in current_path:
                cycle_count += 1
                break
    
    print(total_edges + cycle_count)

if __name__ == "__main__":
    main()