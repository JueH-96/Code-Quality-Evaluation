import collections

def solve():
    n, q = map(int, input().split())
    colors = list(range(1, n + 1))
    
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x, c = query[1], query[2]
            x_index = x - 1
            original_color = colors[x_index]
            if original_color == c:
                continue
            
            reachable_indices = set()
            queue = collections.deque([x_index])
            reachable_indices.add(x_index)
            
            while queue:
                u_index = queue.popleft()
                for v_index in [u_index - 1, u_index + 1]:
                    if 0 <= v_index < n and colors[v_index] == original_color and v_index not in reachable_indices:
                        reachable_indices.add(v_index)
                        queue.append(v_index)
                        
            for index in reachable_indices:
                colors[index] = c
                
        elif query[0] == 2:
            c = query[1]
            count = 0
            for color in colors:
                if color == c:
                    count += 1
            print(count)

if __name__ == '__main__':
    solve()