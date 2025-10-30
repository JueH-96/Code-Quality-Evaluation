def solve():
    n = int(input())
    s = input()
    
    ones = []
    for i in range(n):
        if s[i] == '1':
            ones.append(i)
    
    num_ones = len(ones)
    
    if num_ones == 0:
        print(0)
        return
    
    ans = float('inf')
    
    for i in range(n - num_ones + 1):
        cost = 0
        for j in range(num_ones):
            cost += abs(ones[j] - (i + j))
        ans = min(ans, cost)
    
    print(ans)

solve()