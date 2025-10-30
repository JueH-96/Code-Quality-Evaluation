def solve():
    n = int(input())
    a_list = []
    b_list = []
    for _ in range(n):
        a, b = map(int, input().split())
        a_list.append(a)
        b_list.append(b)
    
    d_list = []
    for i in range(n):
        d_list.append(b_list[i] - a_list[i])
        
    max_d = 0
    for d in d_list:
        max_d = max(max_d, d)
        
    sum_a = sum(a_list)
    
    result = sum_a + max_d
    print(result)

if __name__ == '__main__':
    solve()