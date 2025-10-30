def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    if a == b:
        print(0)
        return

    q = [(a, 0)]
    visited = {tuple(a)}
    
    while q:
        curr_a, steps = q.pop(0)
        
        if curr_a == b:
            print(steps)
            return

        for i in range(n):
            
            # Increment
            next_a = curr_a[:]
            next_a[i] = (next_a[i] + 1) % m
            
            valid = True
            if i > 0 and next_a[i] == next_a[i-1]:
                valid = False
            if i < n - 1 and next_a[i] == next_a[i+1]:
                valid = False
            
            if valid and tuple(next_a) not in visited:
                visited.add(tuple(next_a))
                q.append((next_a, steps + 1))

            # Decrement
            next_a = curr_a[:]
            next_a[i] = (next_a[i] - 1) % m
            
            valid = True
            if i > 0 and next_a[i] == next_a[i-1]:
                valid = False
            if i < n - 1 and next_a[i] == next_a[i+1]:
                valid = False
            
            if valid and tuple(next_a) not in visited:
                visited.add(tuple(next_a))
                q.append((next_a, steps + 1))
    
    print(-1)

solve()