def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    T = input[2]
    
    char_map = {}
    for i in range(N):
        c = S[i]
        d = T[i]
        if c != d:
            if c in char_map:
                if char_map[c] != d:
                    print(-1)
                    return
            else:
                char_map[c] = d
    
    if not char_map:
        print(0)
        return
    
    processed = set()
    valid = True
    
    for c in char_map:
        if c not in processed:
            path = []
            current = c
            while True:
                if current in path:
                    idx = path.index(current)
                    cycle = path[idx:]
                    for node in cycle:
                        if char_map[node] not in cycle:
                            valid = False
                            break
                    if not valid:
                        break
                    for node in path:
                        processed.add(node)
                    break
                if current not in char_map:
                    for node in path:
                        processed.add(node)
                    processed.add(current)
                    break
                if current in processed:
                    for node in path:
                        processed.add(node)
                    break
                path.append(current)
                current = char_map[current]
            if not valid:
                break
    
    if not valid:
        print(-1)
        return
    
    visited_components = set()
    total_ops = 0
    
    for c in char_map:
        if c not in visited_components:
            component = []
            current = c
            in_path = set()
            has_cycle = False
            while True:
                if current in visited_components:
                    break
                if current in in_path:
                    has_cycle = True
                    break
                in_path.add(current)
                component.append(current)
                if current not in char_map:
                    break
                current = char_map[current]
            for node in component:
                visited_components.add(node)
            total_ops += len(component)
            if has_cycle:
                total_ops += 1
    
    print(total_ops)

if __name__ == "__main__":
    main()