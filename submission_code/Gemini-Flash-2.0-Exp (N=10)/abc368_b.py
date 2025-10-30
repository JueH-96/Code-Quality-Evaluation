def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    count = 0
    while True:
        pos_count = 0
        for x in a:
            if x > 0:
                pos_count += 1
        if pos_count <= 1:
            break
            
        a.sort(reverse=True)
        if a[0] > 0 and a[1] > 0:
          a[0] -= 1
          a[1] -= 1
          count += 1
        else:
          break
          
    print(count)

solve()