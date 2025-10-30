def solve():
    n, m = map(int, input().split())
    b = list(map(int, input().split()))
    w = list(map(int, input().split()))

    max_sum = 0
    for i in range(1 << (n + m)):
        chosen_b = []
        chosen_w = []
        current_sum = 0

        for j in range(n):
            if (i >> j) & 1:
                chosen_b.append(b[j])
                current_sum += b[j]
        
        for j in range(m):
            if (i >> (n + j)) & 1:
                chosen_w.append(w[j])
                current_sum += w[j]
        
        if len(chosen_b) >= len(chosen_w):
            max_sum = max(max_sum, current_sum)
    
    print(max_sum)

solve()