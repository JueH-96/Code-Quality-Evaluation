def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    zero_count = 0
    zero_indices = []
    for i in range(n):
        if a[i] == 0:
            zero_count += 1
            zero_indices.append(i)
            
    if zero_count >= 2:
        print(0)
    elif zero_count == 1:
        index_to_increment = zero_indices[0]
        temp_a = list(a)
        temp_a[index_to_increment] += 1
        product = 1
        for digit in temp_a:
            product *= digit
        print(product)
    else:
        min_val = min(a)
        min_index = -1
        for i in range(n):
            if a[i] == min_val:
                min_index = i
                break
        temp_a = list(a)
        temp_a[min_index] += 1
        product = 1
        for digit in temp_a:
            product *= digit
        print(product)

t = int(input())
for _ in range(t):
    solve()