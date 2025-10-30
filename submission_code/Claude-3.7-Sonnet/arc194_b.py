def minimum_cost_to_sort(p):
    n = len(p)
    p = list(p)  # Create a copy of the input permutation to modify
    cost = 0
    
    # Use bubble sort
    for _ in range(n):
        swapped = False
        for i in range(1, n):
            if p[i-1] > p[i]:
                p[i-1], p[i] = p[i], p[i-1]
                cost += i
                swapped = True
        if not swapped:
            break
    
    return cost

n = int(input())
p = list(map(int, input().split()))
print(minimum_cost_to_sort(p))