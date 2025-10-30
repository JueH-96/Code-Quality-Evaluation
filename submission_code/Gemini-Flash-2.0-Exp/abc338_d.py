def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))

    def calculate_distance(start, end, closed_bridge):
        if start == end:
            return 0
        
        dist1 = abs(start - end)
        dist2 = n - abs(start - end)
        
        dist = min(dist1, dist2)
        
        if closed_bridge is not None:
            
            if closed_bridge[0] < closed_bridge[1]:
                if (start <= closed_bridge[0] and end >= closed_bridge[1]) or (end <= closed_bridge[0] and start >= closed_bridge[1]):
                    dist = float('inf')
                else:
                    dist = min(abs(start - end), n - abs(start - end))
            else:
                if (start <= closed_bridge[0] and end >= closed_bridge[1]) or (end <= closed_bridge[0] and start >= closed_bridge[1]):
                    dist = float('inf')
                else:
                    dist = min(abs(start - end), n - abs(start - end))
        
        if dist == float('inf'):
            
            dist1 = 0
            curr = start
            while curr != end:
                
                next_node = (curr % n) + 1
                
                valid = True
                if closed_bridge is not None:
                    if closed_bridge[0] < closed_bridge[1]:
                        if (curr == closed_bridge[0] and next_node == closed_bridge[1]):
                            valid = False
                        if (next_node == closed_bridge[0] and curr == closed_bridge[1]):
                            valid = False
                    else:
                        if (curr == closed_bridge[0] and next_node == closed_bridge[1]):
                            valid = False
                        if (next_node == closed_bridge[0] and curr == closed_bridge[1]):
                            valid = False
                
                if valid:
                    curr = next_node
                    dist1 += 1
                else:
                    dist1 = float('inf')
                    break
            
            dist2 = 0
            curr = start
            while curr != end:
                
                next_node = ((curr - 2 + n) % n) + 1
                
                valid = True
                if closed_bridge is not None:
                    if closed_bridge[0] < closed_bridge[1]:
                        if (curr == closed_bridge[1] and next_node == closed_bridge[0]):
                            valid = False
                        if (next_node == closed_bridge[1] and curr == closed_bridge[0]):
                            valid = False
                    else:
                        if (curr == closed_bridge[1] and next_node == closed_bridge[0]):
                            valid = False
                        if (next_node == closed_bridge[1] and curr == closed_bridge[0]):
                            valid = False
                
                if valid:
                    curr = next_node
                    dist2 += 1
                else:
                    dist2 = float('inf')
                    break
            
            dist = min(dist1, dist2)
        else:
            dist = min(abs(start - end), n - abs(start - end))
        
        return dist

    min_length = float('inf')
    for i in range(1, n + 1):
        closed_bridge = (i, (i % n) + 1)
        
        length = 0
        for j in range(m - 1):
            length += calculate_distance(x[j], x[j+1], closed_bridge)
        min_length = min(min_length, length)

    print(min_length)

solve()