# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))

    x = [val - 1 for val in x]
    
    if k == 0:
        print(*a)
        return

    history = []
    current_a = a[:]
    
    for _ in range(min(k, n + 5)):
        next_a = [0] * n
        for i in range(n):
            next_a[i] = current_a[x[i]]
        
        if next_a in history:
            first_occurrence = history.index(next_a)
            remaining_k = k - (first_occurrence + 1)
            cycle_length = len(history) - first_occurrence
            final_a = history[first_occurrence + (remaining_k % cycle_length)]
            print(*final_a)
            return
        
        history.append(current_a)
        current_a = next_a
    
    print(*current_a)

solve()