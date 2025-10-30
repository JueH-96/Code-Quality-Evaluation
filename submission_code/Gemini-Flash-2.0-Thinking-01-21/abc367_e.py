def solve():
    n, k = map(int, input().split())
    x_input = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    if k == 0:
        print(*(a))
        return
        
    x = [val - 1 for val in x_input] # 0-indexed X
    current_x = list(x)
    result_x = list(range(n)) # identity permutation, 0-indexed

    while k > 0:
        if k % 2 == 1:
            next_result_x = [0] * n
            for i in range(n):
                next_result_x[i] = current_x[result_x[i]]
            result_x = next_result_x
            
        next_current_x = [0] * n
        for i in range(n):
            next_current_x[i] = current_x[current_x[i]]
        current_x = next_current_x
        k //= 2
        
    result_a = [0] * n
    for i in range(n):
        result_a[i] = a[result_x[i]]
        
    print(*(result_a))

if __name__ == '__main__':
    solve()