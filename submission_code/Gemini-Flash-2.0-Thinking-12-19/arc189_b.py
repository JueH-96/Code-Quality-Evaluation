def solve():
    n = int(input())
    x = list(map(int, input().split()))
    
    made_change = True
    while made_change:
        made_change = False
        for i in range(n - 3):
            delta_s = 2 * (x[i] + x[i+3] - x[i+1] - x[i+2])
            if delta_s < 0:
                m = (x[i] + x[i+3]) / 2
                new_x_i_plus_1 = 2 * m - x[i+1]
                new_x_i_plus_2 = 2 * m - x[i+2]
                original_x_i_plus_1 = x[i+1]
                original_x_i_plus_2 = x[i+2]
                x[i+1] = new_x_i_plus_1
                x[i+2] = new_x_i_plus_2
                x.sort()
                made_change = True
                break
            if made_change:
                break
                
    print(sum(x))

if __name__ == '__main__':
    solve()