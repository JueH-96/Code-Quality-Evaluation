def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    has_zero = False
    zero_index = -1
    for i in range(n):
        if a[i] == 0:
            has_zero = True
            zero_index = i
            break
            
    max_product = -1
    
    if has_zero:
        temp_a = list(a)
        temp_a[zero_index] += 1
        current_product = 1
        for digit in temp_a:
            current_product *= digit
        max_product = max(max_product, current_product)
    else:
        min_digit = min(a)
        min_index = -1
        for i in range(n):
            if a[i] == min_digit:
                min_index = i
                break
        temp_a = list(a)
        temp_a[min_index] += 1
        current_product = 1
        for digit in temp_a:
            current_product *= digit
        max_product = max(max_product, current_product)
        
    
    possible_products = []
    for i in range(n):
        temp_a = list(a)
        temp_a[i] += 1
        current_product = 1
        for digit in temp_a:
            current_product *= digit
        possible_products.append(current_product)
        
    print(max(possible_products))

t = int(input())
for _ in range(t):
    solve()