def solve():
    n, m = map(int, input().split())
    x_positions = list(map(int, input().split()))
    a_stones = list(map(int, input().split()))
    
    initial_stones = [0] * n
    total_initial_stones = 0
    for i in range(m):
        pos = x_positions[i]
        stones = a_stones[i]
        initial_stones[pos-1] = stones
        total_initial_stones += stones
        
    if total_initial_stones != n:
        print("-1")
        return
        
    current_stones = list(initial_stones)
    operations_count = 0
    
    for i in range(n - 1):
        if current_stones[i] < 1:
            print("-1")
            return
        stones_to_move = max(0, current_stones[i] - 1)
        current_stones[i] -= stones_to_move
        current_stones[i+1] += stones_to_move
        operations_count += stones_to_move
        
    if current_stones[n-1] == 1:
        print(operations_count)
    else:
        print("-1")

if __name__ == '__main__':
    solve()