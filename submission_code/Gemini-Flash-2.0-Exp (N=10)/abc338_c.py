def solve():
    n = int(input())
    q = list(map(int, input().split()))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    max_servings = 0
    for num_a in range(1001):
        valid = True
        for i in range(n):
            if a[i] * num_a > q[i]:
                valid = False
                break
        if not valid:
            continue
        
        remaining_q = [q[i] - a[i] * num_a for i in range(n)]
        
        min_b_servings = float('inf')
        for i in range(n):
            if b[i] > 0:
                min_b_servings = min(min_b_servings, remaining_q[i] // b[i])
            elif b[i] == 0 and remaining_q[i] > 0:
                continue
            elif b[i] == 0 and remaining_q[i] == 0:
                continue
            elif b[i] == 0 and remaining_q[i] < 0:
                min_b_servings = 0
                break
        
        if min_b_servings == float('inf'):
            min_b_servings = 0
        
        max_servings = max(max_servings, num_a + min_b_servings)

    print(max_servings)

solve()