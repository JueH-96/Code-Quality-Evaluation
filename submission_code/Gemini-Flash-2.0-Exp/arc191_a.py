def solve():
    n, m = map(int, input().split())
    s = list(input())
    t = input()
    
    t_sorted = sorted(list(t), reverse=True)
    
    for k in range(m):
        max_index = -1
        min_digit = 10
        for i in range(n):
            if int(s[i]) < int(t_sorted[k]):
                if max_index == -1 or int(s[i]) < min_digit:
                    max_index = i
                    min_digit = int(s[i])
        
        if max_index != -1:
            s[max_index] = t_sorted[k]
            
    print("".join(s))

solve()