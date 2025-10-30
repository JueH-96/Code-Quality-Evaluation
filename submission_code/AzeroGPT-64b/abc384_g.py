def calculate_abs_diff_sum(a, b, x, y):
    n = len(a)
    m = len(b)
    a.sort()
    b.sort()
    a_left_sum = [0] * (n + 1)
    a_right_sum = [0] * (n + 1)
    b_left_sum = [0] * (m + 1)
    b_right_sum = [0] * (m + 1)
    
    for i in range(1, n + 1):
        a_left_sum[i] = a_left_sum[i - 1] + a[i - 1]
        a_right_sum[i - 1] = a_right_sum[i - 1] + a[i - 1]
    a_right_sum[n] = a_right_sum[n - 1]
    
    for i in range(1, m + 1):
        b_left_sum[i] = b_left_sum[i - 1] + b[i - 1]
        b_right_sum[i - 1] = b_right_sum[i - 1] + b[i - 1]
    b_right_sum[m] = b_right_sum[m - 1]
    
    result = []
    for xi, yi in zip(x, y):
        xi -= 1
        yi -= 1
        count_a_left = xi * a_left_sum[xi + 1] if xi > 0 else 0
        count_a_right = xi * (a_right_sum[0] - a_right_sum[xi + 1])
        count_b_left = yi * b_left_sum[yi + 1] if yi > 0 else 0
        count_b_right = yi * (b_right_sum[0] - b_right_sum[yi + 1])
        
        prefix_diff = a_left_sum[xi + 1] - b_right_sum[yi + 1]
        suffix_diff = b_left_sum[yi + 1] - a_right_sum[xi + 1]
        
        ans = count_a_left + count_a_right + count_b_left + count_b_right + prefix_diff * yi - suffix_diff * xi
        result.append(ans)
    
    return result

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
K = int(input())
XY = [list(map(int, input().split())) for _ in range(K)]

results = calculate_abs_diff_sum(A, B, [xy[0] for xy in XY], [xy[1] for xy in XY])
for res in results:
    print(res)