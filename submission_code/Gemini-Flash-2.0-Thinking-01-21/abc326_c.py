def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    a.sort()
    start_points = set()
    for gift_pos in a:
        start_points.add(gift_pos)
        start_points.add(gift_pos - m)
    
    max_gifts = 0
    test_starts = sorted(list(start_points))
    
    for x in test_starts:
        current_gifts = 0
        for gift_pos in a:
            if x <= gift_pos < x + m:
                current_gifts += 1
        max_gifts = max(max_gifts, current_gifts)
        
    if not test_starts:
        current_gifts = 0
        for gift_pos in a:
            if 0 <= gift_pos < m:
                current_gifts += 1
        max_gifts = max(max_gifts, current_gifts)

    if max_gifts == 0 and n > 0:
        max_gifts = 0
        for x in [0]: # consider origin as a starting point if no critical points found.
            current_gifts = 0
            for gift_pos in a:
                if x <= gift_pos < x + m:
                    current_gifts += 1
            max_gifts = max(max_gifts, current_gifts)

    if n == 0:
        print(0)
    else:
        print(max_gifts)

if __name__ == '__main__':
    solve()