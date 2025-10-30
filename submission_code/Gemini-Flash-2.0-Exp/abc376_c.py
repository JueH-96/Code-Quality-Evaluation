def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    a.sort()
    b.sort()

    ans = float('inf')

    for x in sorted(list(set(a))):
        possible = True
        
        temp_b = sorted(b + [x])
        
        assignment = True
        
        used_boxes = [False] * n
        
        for toy_size in a:
            found_box = False
            for i in range(n):
                if not used_boxes[i] and temp_b[i] >= toy_size:
                    used_boxes[i] = True
                    found_box = True
                    break
            if not found_box:
                assignment = False
                break
        
        if assignment:
            ans = min(ans, x)

    if ans == float('inf'):
        print("-1")
    else:
        print(ans)

solve()