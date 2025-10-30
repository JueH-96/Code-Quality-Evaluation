# YOUR CODE HERE
T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    
    # Find symmetric points
    is_symmetric = [False] * N
    for i in range(N):
        # Check if point i is symmetric for Alice (reflect_A(i) = i)
        if (N - i) % N == i:
            is_symmetric[i] = True
        # Check if point i is symmetric for Bob (reflect_B(i) = i)  
        if (2 * K - i) % N == i:
            is_symmetric[i] = True
    
    # Find orbits using BFS
    visited = [False] * N
    can_color_all = True
    
    for i in range(N):
        if not visited[i]:
            # Check if this orbit has a symmetric point
            has_symmetric = is_symmetric[i]
            queue = [i]
            visited[i] = True
            
            while queue and not has_symmetric:
                u = queue.pop(0)
                
                # Apply Alice's reflection
                v = (N - u) % N
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    if is_symmetric[v]:
                        has_symmetric = True
                
                # Apply Bob's reflection
                v = (2 * K - u) % N
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    if is_symmetric[v]:
                        has_symmetric = True
            
            if not has_symmetric:
                can_color_all = False
                break
    
    print("Yes" if can_color_all else "No")