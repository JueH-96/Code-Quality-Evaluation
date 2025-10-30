from collections import deque

def solve():
    N = int(input())
    graph = []
    for _ in range(N):
        graph.append(input().strip())
    
    # Build adjacency lists for forward and reverse edges
    forward = [[] for _ in range(N)]
    reverse = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if graph[i][j] != '-':
                forward[i].append((j, graph[i][j]))
                reverse[j].append((i, graph[i][j]))
    
    # Find shortest palindromic path from each i to each j
    result = [[-1] * N for _ in range(N)]
    
    # Empty path is a palindrome
    for i in range(N):
        result[i][i] = 0
    
    # BFS-like approach for finding shortest palindromic paths
    # We'll use a queue with states (start, end, length)
    for start in range(N):
        for end in range(N):
            if result[start][end] != -1:
                continue
                
            # Try to find shortest palindromic path
            visited = set()
            queue = deque()
            
            # Check single edge palindromes
            for next_v, label in forward[start]:
                if next_v == end:
                    result[start][end] = 1
                    break
            
            if result[start][end] != -1:
                continue
            
            # Initialize queue with all possible starting states
            # For even length palindromes
            for v1, label1 in forward[start]:
                for v2, label2 in reverse[end]:
                    if label1 == label2:
                        if v1 == v2:
                            result[start][end] = 2
                            break
                        queue.append((v1, v2, 2))
                if result[start][end] != -1:
                    break
            
            if result[start][end] != -1:
                continue
            
            # For odd length palindromes (center vertex)
            for center in range(N):
                for v1, label1 in reverse[center]:
                    if v1 == start:
                        for v2, label2 in forward[center]:
                            if v2 == end and label1 == label2:
                                result[start][end] = 3
                                break
                    if result[start][end] != -1:
                        break
                if result[start][end] != -1:
                    break
            
            if result[start][end] != -1:
                continue
            
            # BFS for longer palindromes
            while queue and result[start][end] == -1:
                curr_v1, curr_v2, length = queue.popleft()
                
                if (curr_v1, curr_v2, length) in visited:
                    continue
                visited.add((curr_v1, curr_v2, length))
                
                # Try extending the palindrome
                for next_v1, label1 in forward[curr_v1]:
                    for next_v2, label2 in reverse[curr_v2]:
                        if label1 == label2:
                            if next_v1 == next_v2:
                                result[start][end] = length + 2
                                break
                            elif (next_v1, next_v2, length + 2) not in visited:
                                queue.append((next_v1, next_v2, length + 2))
                    if result[start][end] != -1:
                        break
                
                # Also check odd length extensions
                if result[start][end] == -1:
                    for center in range(N):
                        found = False
                        for v1, label1 in reverse[center]:
                            if v1 == curr_v1:
                                for v2, label2 in forward[center]:
                                    if v2 == curr_v2 and label1 == label2:
                                        result[start][end] = length + 3
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
    
    # Print results
    for row in result:
        print(' '.join(map(str, row)))

solve()