def solve():
    n, q = map(int, input().split())
    
    left_hand = 1
    right_hand = 2
    total_ops = 0
    
    for _ in range(q):
        h, t = input().split()
        t = int(t)
        
        if h == 'L':
            if left_hand != t:
                dist1 = abs(left_hand - t)
                dist2 = n - dist1
                
                if t > left_hand:
                    if right_hand > left_hand and right_hand < t:
                        total_ops += dist2
                    else:
                        total_ops += dist1
                else:
                    if right_hand < left_hand and right_hand > t:
                        total_ops += dist2
                    else:
                        total_ops += dist1
                
                left_hand = t
        else:
            if right_hand != t:
                dist1 = abs(right_hand - t)
                dist2 = n - dist1
                
                if t > right_hand:
                    if left_hand > right_hand and left_hand < t:
                        total_ops += dist2
                    else:
                        total_ops += dist1
                else:
                    if left_hand < right_hand and left_hand > t:
                        total_ops += dist2
                    else:
                        total_ops += dist1
                
                right_hand = t
    
    print(total_ops)

solve()