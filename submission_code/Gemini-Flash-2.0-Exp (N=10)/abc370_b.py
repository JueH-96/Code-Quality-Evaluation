def solve():
    n = int(input())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))
    
    current_element = 1
    for i in range(1, n + 1):
        if current_element >= i:
            current_element = a[current_element - 1][i - 1]
        else:
            current_element = a[i - 1][current_element - 1]
    print(current_element)

solve()