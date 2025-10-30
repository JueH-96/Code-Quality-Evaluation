def solve():
    n, q = map(int, input().split())
    s_list = list(input())
    
    for _ in range(q):
        x_i, c_i = input().split()
        x_i = int(x_i)
        s_list[x_i - 1] = c_i
        
        abc_count = 0
        for i in range(n - 2):
            if s_list[i] == 'A' and s_list[i+1] == 'B' and s_list[i+2] == 'C':
                abc_count += 1
        print(abc_count)

if __name__ == '__main__':
    solve()