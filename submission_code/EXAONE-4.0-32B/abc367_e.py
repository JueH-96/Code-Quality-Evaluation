import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    k = int(data[1])
    x_list = list(map(int, data[2:2+n]))
    a_list = list(map(int, data[2+n:2+2*n]))
    
    if k == 0:
        print(" ".join(map(str, a_list)))
        return
        
    next_arr = [x - 1 for x in x_list]
    
    LOG = 60
    dp = [[0] * n for _ in range(LOG+1)]
    
    for i in range(n):
        dp[0][i] = next_arr[i]
        
    for j in range(1, LOG+1):
        for i in range(n):
            dp[j][i] = dp[j-1][dp[j-1][i]]
            
    target = [0] * n
    for i in range(n):
        cur = i
        for j in range(LOG+1):
            if (k >> j) & 1:
                cur = dp[j][cur]
        target[i] = cur
        
    result_arr = [a_list[target[i]] for i in range(n)]
    print(" ".join(map(str, result_arr)))
    
if __name__ == "__main__":
    main()