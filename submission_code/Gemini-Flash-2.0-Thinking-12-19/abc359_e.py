def solve():
    n = int(input())
    h = list(map(int, input().split()))
    if n == 0:
        print()
        return
    
    op_counts = [0] * n
    op_counts[0] = h[0] + 1
    
    for i in range(1, n):
        prev_op_count = op_counts[i-1]
        d_i = 0
        current_index = i + 1
        if current_index % 2 == 0:
            d_i = h[i]
        elif current_index % 4 == 3:
            d_i = 2 * h[i]
        elif current_index % 4 == 1:
            d_i = 2 * (h[i] + 1)
        else:
            # This case should not happen for i >= 2 (current_index >= 3)
            # For i=1 (current_index=2), i is even, so first condition applies.
            # For i=2 (current_index=3), i % 4 == 3, second condition.
            # For i=3 (current_index=4), i is even, first condition.
            # For i=4 (current_index=5), i % 4 == 1, third condition.
            # For i=5 (current_index=6), i is even, first condition.
            # So, for i>=2, cases are covered.
            pass
            
        op_counts[i] = prev_op_count + d_i
        
    print(*(op_counts))

if __name__ == '__main__':
    solve()