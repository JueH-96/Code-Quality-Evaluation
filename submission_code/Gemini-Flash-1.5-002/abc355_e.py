# YOUR CODE HERE
N, L, R = map(int, input().split())
total_sum = 0
while L <= R:
    i = 0
    while (1 << (i + 1)) * (L // (1 << i) + 1) <= (1 << N):
        i += 1
    i -= 1
    j = L // (1 << i)
    l = j * (1 << i)
    r = (j + 1) * (1 << i) - 1
    print(f'? {i} {j}')
    query_sum = int(input())
    if query_sum == -1:
        break
    if l <= R:
        if r <= R:
            total_sum = (total_sum + query_sum) % 100
            L = r + 1
        else:
            print(f'? {i} {R // (1 << i)}')
            query_sum2 = int(input())
            if query_sum2 == -1:
                break
            total_sum = (total_sum + query_sum2) % 100
            L = R + 1
    else:
        break
print(f'! {total_sum}')