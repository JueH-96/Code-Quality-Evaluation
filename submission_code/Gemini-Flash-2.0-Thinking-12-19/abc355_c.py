def solve():
    n, t = map(int, input().split())
    a_values = list(map(int, input().split()))
    
    marked = [[False for _ in range(n)] for _ in range(n)]
    row_counts = [0] * n
    col_counts = [0] * n
    main_diag_count = 0
    anti_diag_count = 0
    
    for turn in range(1, t + 1):
        current_a = a_values[turn-1]
        r = (current_a - 1) // n
        c = (current_a - 1) % n
        marked[r][c] = True
        row_counts[r] += 1
        col_counts[c] += 1
        if r == c:
            main_diag_count += 1
        if r + c == n - 1:
            anti_diag_count += 1
            
        bingo_achieved = False
        if row_counts[r] == n:
            bingo_achieved = True
        if not bingo_achieved and col_counts[c] == n:
            bingo_achieved = True
        if not bingo_achieved and main_diag_count == n:
            bingo_achieved = True
        if not bingo_achieved and anti_diag_count == n:
            bingo_achieved = True
            
        if bingo_achieved:
            return turn
            
    return -1

if __name__ == '__main__':
    result = solve()
    print(result)