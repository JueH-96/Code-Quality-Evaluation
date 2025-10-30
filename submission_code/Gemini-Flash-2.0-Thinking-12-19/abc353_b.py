def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    empty_seats = k
    attraction_starts = 0
    
    for group_size in a:
        if empty_seats < group_size:
            attraction_starts += 1
            empty_seats = k
        empty_seats -= group_size
        
    attraction_starts += 1
    print(attraction_starts)

if __name__ == '__main__':
    solve()