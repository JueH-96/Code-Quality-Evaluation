def solve():
    n = int(input())
    giants = []
    for _ in range(n):
        a, b = map(int, input().split())
        giants.append({'a': a, 'b': b})
    
    max_c = -float('inf')
    max_c_index = -1
    for i in range(n):
        c_i = giants[i]['b'] - giants[i]['a']
        if c_i > max_c:
            max_c = c_i
            max_c_index = i
            
    sum_a = 0
    for i in range(n):
        sum_a += giants[i]['a']
        
    max_head_height = sum_a + max_c
    print(max_head_height)

if __name__ == '__main__':
    solve()