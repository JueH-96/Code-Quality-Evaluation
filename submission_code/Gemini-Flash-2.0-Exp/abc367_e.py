def solve():
    n, k = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))

    x = [val - 1 for val in x]

    if k == 0:
        print(*a)
        return

    history = [a]
    
    for _ in range(min(k, n + 5)):
        b = [0] * n
        for i in range(n):
            b[i] = a[x[i]]
        a = b
        
        if a in history:
            first_occurrence = history.index(a)
            cycle_length = len(history) - first_occurrence
            remaining_ops = (k - len(history)) % cycle_length
            a = history[first_occurrence + remaining_ops]
            print(*a)
            return
        
        history.append(a)

    print(*a)

solve()