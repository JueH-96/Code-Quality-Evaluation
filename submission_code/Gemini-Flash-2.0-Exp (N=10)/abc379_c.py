def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))

    total_stones = sum(a)
    if total_stones != n:
        print(-1)
        return

    stones = {}
    for i in range(m):
        stones[x[i]] = a[i]
    
    
    operations = 0
    current_stones = 0
    
    
    for i in range(1, n + 1):
      if i in stones:
        current_stones += stones[i]
      
      current_stones -= 1
      
      if current_stones < 0:
          print(-1)
          return
      
      operations += current_stones
    
    print(operations)
    

solve()