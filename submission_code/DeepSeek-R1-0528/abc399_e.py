import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0].strip())
    S = data[1].strip()
    T = data[2].strip()
    
    if S == T:
        print(0)
        return
        
    mapping_target = defaultdict(set)
    for i in range(n):
        s_char = S[i]
        t_char = T[i]
        mapping_target[s_char].add(t_char)
        
    for c in mapping_target:
        if len(mapping_target[c]) > 1:
            print(-1)
            return
            
    mapping_dict = {}
    V = set(S) | set(T)
    
    for c in mapping_target:
        d = next(iter(mapping_target[c]))
        if c != d:
            mapping_dict[c] = d
            
    if not mapping_dict:
        print(0)
        return
        
    visited = set()
    cycles = 0
    for node in mapping_dict:
        if node not in visited:
            stack = []
            cur = node
            while cur in mapping_dict and cur not in visited:
                visited.add(cur)
                stack.append(cur)
                cur = mapping_dict[cur]
                
            if cur in stack:
                cycles += 1
                
    if cycles > 0 and len(V) == 26:
        print(-1)
    else:
        total_operations = len(mapping_dict) + cycles
        print(total_operations)

if __name__ == "__main__":
    main()