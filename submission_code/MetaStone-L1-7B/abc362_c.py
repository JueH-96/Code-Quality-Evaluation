n = int(input())
pairs = [tuple(map(int, input().split())) for _ in range(n)]

sum_L = sum(L for L, R in pairs)
sum_R = sum(R for L, R in pairs)

if sum_L > 0 or sum_R < 0:
    print("No")
else:
    delta = 0 - sum_L
    possible_increases = [(R - L, i) for i, (L, R) in enumerate(pairs)]
    possible_increases.sort(reverse=True, key=lambda x: x[0])
    
    X = [L for L, R in pairs]
    current_delta = delta
    
    for inc, i in possible_increases:
        if current_delta <= 0:
            break
        add = min(inc, current_delta)
        X[i] += add
        current_delta -= add
    
    print("Yes")
    print(' '.join(map(str, X)))