def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    empty_seats = k
    starts = 0
    queue = a
    
    while queue:
        if not queue:
            break
        
        if empty_seats < queue[0]:
            starts += 1
            empty_seats = k
        else:
            empty_seats -= queue[0]
            queue.pop(0)
            
    if queue:
      starts +=1
    
    print(starts)

solve()