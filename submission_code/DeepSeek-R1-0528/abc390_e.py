import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    X_global = int(next(it))
    foods = []
    for i in range(n):
        v = int(next(it))
        a = int(next(it))
        c = int(next(it))
        foods.append((v, a, c))
    
    list1 = []
    list2 = []
    list3 = []
    for (v, a, c) in foods:
        if v == 1:
            list1.append((a, c))
        elif v == 2:
            list2.append((a, c))
        else:
            list3.append((a, c))
            
    def compute_dp(food_list, X):
        dp = [0] * (X+1)
        for (a, c) in food_list:
            for j in range(X, c-1, -1):
                if dp[j] < dp[j-c] + a:
                    dp[j] = dp[j-c] + a
        for j in range(1, X+1):
            if dp[j] < dp[j-1]:
                dp[j] = dp[j-1]
        return dp

    dp1 = compute_dp(list1, X_global)
    dp2 = compute_dp(list2, X_global)
    dp3 = compute_dp(list3, X_global)
    
    M1 = dp1[X_global]
    M2 = dp2[X_global]
    M3 = dp3[X_global]
    M = min(M1, M2, M3)
    
    low, high = 0, M+1
    while low < high:
        mid = (low + high) // 2
        total_cal = 0
        valid = True
        for d in [dp1, dp2, dp3]:
            if mid == 0:
                cal_need = 0
            else:
                if d[X_global] < mid:
                    valid = False
                    break
                lo2, hi2 = 0, X_global
                while lo2 < hi2:
                    m2 = (lo2 + hi2) // 2
                    if d[m2] >= mid:
                        hi2 = m2
                    else:
                        lo2 = m2 + 1
                cal_need = lo2
            total_cal += cal_need
            if total_cal > X_global:
                valid = False
                break
        if valid:
            low = mid + 1
        else:
            high = mid
    print(low-1)

if __name__ == "__main__":
    main()