def solve():
    n = int(input())
    s = input()
    a = [int(c) for c in s]
    
    total_sum = 0
    for i in range(n):
        for j in range(i, n):
            current_val = a[i]
            for k in range(i + 1, j + 1):
                current_val = 1 - (current_val & a[k])
            total_sum += current_val
    print(total_sum)

solve()