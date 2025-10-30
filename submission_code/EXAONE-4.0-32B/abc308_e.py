def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    A = list(map(int, data[1].split()))
    S = data[2].strip()
    
    mex_table = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                s = {a, b, c}
                m = 0
                while m in s:
                    m += 1
                mex_table[a][b][c] = m
    
    left_counts = [None] * n
    cnt_M = [0, 0, 0]
    for i in range(n):
        if S[i] == 'M':
            val = A[i]
            cnt_M[val] += 1
        elif S[i] == 'E':
            left_counts[i] = cnt_M.copy()
    
    right_counts = [None] * n
    cnt_X = [0, 0, 0]
    for i in range(n-1, -1, -1):
        if S[i] == 'X':
            val = A[i]
            cnt_X[val] += 1
        elif S[i] == 'E':
            right_counts[i] = cnt_X.copy()
    
    ans = 0
    for i in range(n):
        if S[i] == 'E':
            b_val = A[i]
            left_arr = left_counts[i]
            right_arr = right_counts[i]
            for a_val in range(3):
                for c_val in range(3):
                    count_left = left_arr[a_val]
                    count_right = right_arr[c_val]
                    count_triples = count_left * count_right
                    m = mex_table[a_val][b_val][c_val]
                    ans += count_triples * m
    print(ans)

if __name__ == "__main__":
    main()