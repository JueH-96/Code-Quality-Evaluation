import math

def solve():
    n = int(input())
    abc_values = []
    for _ in range(n):
        abc_values.append(list(map(int, input().split())))
    
    for i in range(n):
        if abc_values[i][2] <= abc_values[i][1]:
            print(0)
            return
            
    y_prime_values = []
    for i in range(n):
        y_prime_values.append((abc_values[i][2] - abc_values[i][0]) / abc_values[i][1])
    
    y_prime_max = max(y_prime_values)
    y_max = max(0, math.ceil(y_prime_max) - 1)
    
    total_count = 0
    for y in range(1, y_max + 1):
        v_values = []
        for i in range(n):
            v_values.append((abc_values[i][2] - abc_values[i][1] * y) / abc_values[i][0])
        v_y = min(v_values)
        x_count = max(0, math.floor(v_y - 1e-9))
        total_count += x_count
        
    print(total_count)

t = int(input())
for _ in range(t):
    solve()