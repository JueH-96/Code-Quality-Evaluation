def solve():
    n, k = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    x = [val - 1 for val in x]

    if k == 0:
        print(*a)
        return
    
    history = []
    current_a = list(a)
    
    for _ in range(min(k, 2 * n)):
        history.append(list(current_a))
        next_a = [0] * n
        for i in range(n):
            next_a[i] = current_a[x[i]]
        current_a = next_a
        
        if current_a in history:
            first_occurrence = history.index(current_a)
            cycle_len = len(history) - first_occurrence
            remaining_ops = (k - len(history)) % cycle_len
            print(*history[first_occurrence + remaining_ops])
            return
    
    print(*current_a)

solve()