def solve():
    n = int(input())
    q = list(map(int, input().split()))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    max_servings = 0
    for num_a in range(1001):
        possible = True
        remaining_q = list(q)
        for i in range(n):
            remaining_q[i] -= num_a * a[i]
            if remaining_q[i] < 0:
                possible = False
                break
        
        if not possible:
            continue

        num_b = float('inf')
        for i in range(n):
            if b[i] > 0:
                num_b = min(num_b, remaining_q[i] // b[i])
        
        if num_b == float('inf'):
            num_b = 0
        
        max_servings = max(max_servings, num_a + num_b)

    
    for num_b in range(1001):
        possible = True
        remaining_q = list(q)
        for i in range(n):
            remaining_q[i] -= num_b * b[i]
            if remaining_q[i] < 0:
                possible = False
                break
        
        if not possible:
            continue

        num_a = float('inf')
        for i in range(n):
            if a[i] > 0:
                num_a = min(num_a, remaining_q[i] // a[i])
        
        if num_a == float('inf'):
            num_a = 0
        
        max_servings = max(max_servings, num_a + num_b)

    print(max_servings)

solve()