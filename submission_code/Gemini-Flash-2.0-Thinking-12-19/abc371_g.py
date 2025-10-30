def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    visited = [False] * (n + 1)
    cycles = []
    for i in range(1, n + 1):
        if not visited[i]:
            cycle_indices = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle_indices.append(current)
                current = p[current-1]
            cycles.append(cycle_indices)
            
    result_a = list(a)
    
    for cycle_indices in cycles:
        cycle_values = []
        for index in cycle_indices:
            cycle_values.append(a[index-1])
        
        best_shift_values = cycle_values
        for _ in range(len(cycle_values)):
            shifted_values = cycle_values[1:] + [cycle_values[0]]
            if shifted_values < best_shift_values:
                best_shift_values = shifted_values
            cycle_values = shifted_values
            
        for i in range(len(cycle_indices)):
            result_a[cycle_indices[i]-1] = best_shift_values[i]
            
    print(*(result_a))

if __name__ == '__main__':
    solve()