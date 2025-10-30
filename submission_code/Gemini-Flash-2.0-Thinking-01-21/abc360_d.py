def solve():
    n, t = map(int, input().split())
    s = input()
    x = list(map(int, input().split()))
    
    pass_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dir_i = 1 if s[i] == '1' else -1
            dir_j = 1 if s[j] == '1' else -1
            
            if dir_i == 1 and dir_j == -1:
                if x[i] <= x[j] and x[j] - x[i] <= 2 * t:
                    pass_count += 1
            elif dir_i == -1 and dir_j == 1:
                if x[i] >= x[j] and x[i] - x[j] <= 2 * t:
                    pass_count += 1
                    
    print(pass_count)

if __name__ == '__main__':
    solve()