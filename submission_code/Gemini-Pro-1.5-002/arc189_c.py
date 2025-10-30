def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))
    
    x -= 1
    p = [i - 1 for i in p]
    q = [i - 1 for i in q]

    initial_a = list(a)
    initial_b = list(b)

    q_red = [[] for _ in range(n)]
    q_blue = [[] for _ in range(n)]
    for i in range(n):
        q_red[p[i]].append(i)
        q_blue[q[i]].append(i)
    
    def check(moves):
        a = list(initial_a)
        b = list(initial_b)
        
        moved = [False] * n
        
        for move in moves:
            moved[move] = True
            
        for move in moves:
            red_count = a[move]
            blue_count = b[move]
            a[move] = 0
            b[move] = 0
            a[p[move]] += red_count
            b[q[move]] += blue_count
            
        for i in range(n):
            if i != x and (a[i] > 0 or b[i] > 0):
                return False
        return True

    def get_moves(start_nodes_red, start_nodes_blue):
        moves = set()
        q_r = list(start_nodes_red)
        q_b = list(start_nodes_blue)
        
        while q_r or q_b:
            if q_r:
                curr = q_r.pop()
                if curr not in moves:
                    moves.add(curr)
                    for neighbor in q_red[curr]:
                        if neighbor not in moves and initial_a[neighbor] > 0:
                            q_r.append(neighbor)
                    for neighbor in q_blue[curr]:
                        if neighbor not in moves and initial_b[neighbor] > 0:
                            q_b.append(neighbor)
            if q_b:
                curr = q_b.pop()
                if curr not in moves:
                    moves.add(curr)
                    for neighbor in q_red[curr]:
                        if neighbor not in moves and initial_a[neighbor] > 0:
                            q_r.append(neighbor)
                    for neighbor in q_blue[curr]:
                        if neighbor not in moves and initial_b[neighbor] > 0:
                            q_b.append(neighbor)
        return list(moves)

    start_nodes_red = []
    start_nodes_blue = []
    for i in range(n):
        if i != x and (initial_a[i] > 0 or initial_b[i] > 0):
            if initial_a[i] > 0:
                start_nodes_red.append(i)
            if initial_b[i] > 0:
                start_nodes_blue.append(i)
                
    if not start_nodes_red and not start_nodes_blue:
        print(0)
        return

    moves = get_moves(start_nodes_red, start_nodes_blue)
    
    if check(moves):
        print(len(moves))
    else:
        print(-1)

solve()