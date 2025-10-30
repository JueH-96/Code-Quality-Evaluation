def solve():
    n = int(input())
    p = list(map(int, input().split()))

    def check(arr):
        for i in range(len(arr)):
            if arr[i] != i + 1:
                return False
        return True

    if check(p):
        print(0)
        return

    def operate(arr, k):
        new_arr = arr[:]
        if k >= 2:
            new_arr[:k-1] = sorted(new_arr[:k-1])
        if k <= n - 1:
            new_arr[k:] = sorted(new_arr[k:])
        return new_arr

    
    q = [(p, 0)]
    visited = {tuple(p)}
    
    while q:
        curr_p, steps = q.pop(0)
        
        if check(curr_p):
            print(steps)
            return
        
        for k in range(1, n + 1):
            next_p = operate(curr_p, k)
            if tuple(next_p) not in visited:
                q.append((next_p, steps + 1))
                visited.add(tuple(next_p))

t = int(input())
for _ in range(t):
    solve()