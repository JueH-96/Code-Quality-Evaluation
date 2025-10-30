def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    empty_seats = k
    groups_queue = a
    attraction_starts = 0
    
    while groups_queue:
        current_group_size = groups_queue[0]
        if empty_seats < current_group_size:
            attraction_starts += 1
            empty_seats = k
        else:
            empty_seats -= current_group_size
            groups_queue.pop(0)
            
    attraction_starts += 1
    print(attraction_starts)

if __name__ == '__main__':
    solve()