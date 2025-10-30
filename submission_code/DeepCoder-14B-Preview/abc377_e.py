def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    P = list(map(int, input[idx:idx+n]))
    idx += n
    
    # Convert to 1-based index
    P = [0] + P
    
    visited = [False] * (n + 1)
    result = [0] * (n + 1)
    
    for i in range(1, n + 1):
        if not visited[i]:
            cycle = []
            current = i
            while True:
                cycle.append(current)
                visited[current] = True
                current = P[current]
                if current == i:
                    break
            L = len(cycle)
            if L == 0:
                continue
            # Compute m = 2^K mod L
            if L == 1:
                m = 0
            else:
                m = pow(2, K, L)
            # Create a position map
            pos = {element: idx for idx, element in enumerate(cycle)}
            for element in cycle:
                idx_in_cycle = pos[element]
                new_idx = (idx_in_cycle + m) % L
                result[element] = cycle[new_idx]
    
    # Print the result from 1 to n
    print(' '.join(map(str, result[1:n+1])))

if __name__ == '__main__':
    main()