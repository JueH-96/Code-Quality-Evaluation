def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    p = [x - 1 for x in p]

    if k == 0:
        print(*[x + 1 for x in p])
        return

    
    history = []
    current_p = p[:]
    
    
    for _ in range(min(k, 2*n)):
        history.append(current_p[:])
        next_p = [0] * n
        for i in range(n):
            next_p[i] = current_p[current_p[i]]
        current_p = next_p
        
        if current_p in history:
            first_occurrence = history.index(current_p)
            cycle_len = len(history) - first_occurrence
            remaining_ops = (k - len(history)) % cycle_len
            
            print(*[x + 1 for x in history[first_occurrence + remaining_ops]])
            return
    
    print(*[x + 1 for x in current_p])

solve()