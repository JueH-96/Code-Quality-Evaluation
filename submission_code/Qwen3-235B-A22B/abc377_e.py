import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    visited = [False] * (N + 1)
    cycles = []
    
    for i in range(1, N + 1):
        if not visited[i]:
            current = i
            new_cycle = []
            while not visited[current]:
                visited[current] = True
                new_cycle.append(current)
                current = P[current - 1]
            cycles.append(new_cycle)
    
    element_to_cycle = [0] * (N + 1)
    element_to_pos = [0] * (N + 1)
    
    for cycle_idx, cycle in enumerate(cycles):
        for pos, elem in enumerate(cycle):
            element_to_cycle[elem] = cycle_idx
            element_to_pos[elem] = pos
    
    result = [0] * N
    for i in range(1, N + 1):
        cycle_idx = element_to_cycle[i]
        cycle = cycles[cycle_idx]
        L = len(cycle)
        e = pow(2, K, L)
        pos_in_cycle = element_to_pos[i]
        new_pos = (pos_in_cycle + e) % L
        new_elem = cycle[new_pos]
        result[i - 1] = new_elem
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()