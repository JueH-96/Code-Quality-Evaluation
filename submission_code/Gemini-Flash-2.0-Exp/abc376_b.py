def solve():
    n, q = map(int, input().split())
    
    left = 1
    right = 2
    
    total_moves = 0
    
    for _ in range(q):
        h, t = input().split()
        t = int(t)
        
        if h == 'L':
            
            dist1 = abs(left - t)
            dist2 = n - dist1
            
            moves = min(dist1, dist2)
            
            if dist1 <= dist2:
                if left < t:
                    for _ in range(moves):
                        left = (left % n) + 1
                        if left == right:
                            left = (left % n) + 1
                else:
                    for _ in range(moves):
                        left = (left - 2 + n) % n + 1
                        if left == right:
                            left = (left - 2 + n) % n + 1
            else:
                if left < t:
                    for _ in range(moves):
                        left = (left + 1) % n + 1
                        if left == right:
                            left = (left + 1) % n + 1
                else:
                    for _ in range(moves):
                        left = (left - 1 + n) % n + 1
                        if left == right:
                            left = (left - 1 + n) % n + 1
            
            left = t
            total_moves += moves
            
        else:
            dist1 = abs(right - t)
            dist2 = n - dist1
            
            moves = min(dist1, dist2)
            
            if dist1 <= dist2:
                if right < t:
                    for _ in range(moves):
                        right = (right % n) + 1
                        if right == left:
                            right = (right % n) + 1
                else:
                    for _ in range(moves):
                        right = (right - 2 + n) % n + 1
                        if right == left:
                            right = (right - 2 + n) % n + 1
            else:
                if right < t:
                    for _ in range(moves):
                        right = (right + 1) % n + 1
                        if right == left:
                            right = (right + 1) % n + 1
                else:
                    for _ in range(moves):
                        right = (right - 1 + n) % n + 1
                        if right == left:
                            right = (right - 1 + n) % n + 1
            
            right = t
            total_moves += moves
            
    print(total_moves)

solve()