import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    X = list(map(int, data[2:2+m]))
    X = [x-1 for x in X]
    
    S = 0
    for i in range(m-1):
        a = X[i]
        b = X[i+1]
        if a <= b:
            d_val = b - a
        else:
            d_val = b + n - a
        S += d_val
        
    diff_cnt = [0] * (n+1)
    diff_sd = [0] * (n+1)
    
    for i in range(m-1):
        a = X[i]
        b = X[i+1]
        if a == b:
            continue
        if a < b:
            d_val = b - a
            diff_cnt[a] += 1
            if b < n:
                diff_cnt[b] -= 1
            diff_sd[a] += d_val
            if b < n:
                diff_sd[b] -= d_val
        else:
            d_val = b + n - a
            diff_cnt[a] += 1
            diff_cnt[n] -= 1
            diff_sd[a] += d_val
            diff_sd[n] -= d_val
            
            if b > 0:
                diff_cnt[0] += 1
                diff_cnt[b] -= 1
                diff_sd[0] += d_val
                diff_sd[b] -= d_val
                
    cnt_arr = [0] * n
    sd_arr = [0] * n
    cur_cnt = 0
    cur_sd = 0
    for i in range(n):
        cur_cnt += diff_cnt[i]
        cur_sd += diff_sd[i]
        cnt_arr[i] = cur_cnt
        sd_arr[i] = cur_sd
        
    ans = float('inf')
    for r in range(n):
        T_r = S + n * cnt_arr[r] - 2 * sd_arr[r]
        if T_r < ans:
            ans = T_r
            
    print(ans)

if __name__ == "__main__":
    main()