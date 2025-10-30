def solve():
    n, t = map(int, input().split())
    a = list(map(int, input().split()))
    
    row_marked_count = [0] * n
    col_marked_count = [0] * n
    main_diag_marked_count = 0
    anti_diag_marked_count = 0
    
    for turn_number in range(1, t + 1):
        announced_number = a[turn_number - 1]
        row_index_0based = (announced_number - 1) // n
        col_index_0based = (announced_number - 1) % n
        
        row_marked_count[row_index_0based] += 1
        col_marked_count[col_index_0based] += 1
        if row_index_0based == col_index_0based:
            main_diag_marked_count += 1
        if row_index_0based + col_index_0based == n - 1:
            anti_diag_marked_count += 1
            
        bingo_achieved = False
        if row_marked_count[row_index_0based] == n:
            bingo_achieved = True
        if col_marked_count[col_index_0based] == n:
            bingo_achieved = True
        if row_index_0based == col_index_0based and main_diag_marked_count == n:
            bingo_achieved = True
        if row_index_0based + col_index_0based == n - 1 and anti_diag_marked_count == n:
            bingo_achieved = True
            
        if bingo_achieved:
            print(turn_number)
            return
            
    print("-1")

if __name__ == '__main__':
    solve()