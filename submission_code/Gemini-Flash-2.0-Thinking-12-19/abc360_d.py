def solve():
    n, t = map(int, input().split())
    s = input()
    x = list(map(int, input().split()))
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dir_i = int(s[i])
            dir_j = int(s[j])
            pos_i = x[i]
            pos_j = x[j]
            
            if dir_i == 1 and dir_j == 0:
                if pos_i < pos_j and pos_j <= pos_i + 2 * t:
                    count += 1
            elif dir_i == 0 and dir_j == 1:
                if pos_i > pos_j and pos_j >= pos_i - 2 * t:
                    count += 1
                    
    print(count)

if __name__ == '__main__':
    solve()