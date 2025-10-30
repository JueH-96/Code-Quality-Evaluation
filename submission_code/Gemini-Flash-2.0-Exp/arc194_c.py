def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    c = list(map(int, input().split()))

    cost = 0
    for i in range(n):
        if a[i] != b[i]:
            temp_a = a[:]
            temp_a[i] = 1 - temp_a[i]
            
            current_cost = 0
            for k in range(n):
                current_cost += temp_a[k] * c[k]
            
            cost += current_cost
            a = temp_a
            
    print(cost)

solve()