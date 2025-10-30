def solve():
    n, q = map(int, input().split())
    pattern = [input() for _ in range(n)]
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))
    
    black_count_nxn = 0
    for i in range(n):
        for j in range(n):
            if pattern[i][j] == 'B':
                black_count_nxn += 1
                
    black_count_row = [0] * n
    for i in range(n):
        for j in range(n):
            if pattern[i][j] == 'B':
                black_count_row[i] += 1
                
    black_count_col = [0] * n
    for j in range(n):
        for i in range(n):
            if pattern[i][j] == 'B':
                black_count_col[j] += 1
                
    results = []
    for query in queries:
        a, b, c, d = query
        num_full_rows = (c - a + 1) // n
        num_full_cols = (d - b + 1) // n
        rem_rows = (c - a + 1) % n
        rem_cols = (d - b + 1) % n
        
        count1 = black_count_nxn * num_full_rows * num_full_cols
        count2 = 0
        if rem_rows > 0 and num_full_cols > 0:
            start_rem_row = a + num_full_rows * n
            for i in range(start_rem_row, c + 1):
                for j in range(b, b + num_full_cols * n):
                    if pattern[i % n][j % n] == 'B':
                        count2 += 1
                        
        count3 = 0
        if num_full_rows > 0 and rem_cols > 0:
            start_rem_col = b + num_full_cols * n
            for i in range(a, a + num_full_rows * n):
                for j in range(start_rem_col, d + 1):
                    if pattern[i % n][j % n] == 'B':
                        count3 += 1
                        
        count4 = 0
        if rem_rows > 0 and rem_cols > 0:
            start_rem_row = a + num_full_rows * n
            start_rem_col = b + num_full_cols * n
            for i in range(start_rem_row, c + 1):
                for j in range(start_rem_col, d + 1):
                    if pattern[i % n][j % n] == 'B':
                        count4 += 1
                        
        total_count = count1 + count2 + count3 + count4
        results.append(total_count)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()