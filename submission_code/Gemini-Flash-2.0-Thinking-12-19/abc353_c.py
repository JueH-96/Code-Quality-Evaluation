def solve():
    n = int(input())
    a = list(map(int, input().split()))
    m = 10**8
    a_sorted = sorted(a)
    s_prime = sum(a_sorted)
    count_c = 0
    for i in range(n - 1):
        target = m - a_sorted[i]
        l, r = i + 1, n - 1
        j_0 = -1
        while l <= r:
            mid = (l + r) // 2
            if a_sorted[mid] >= target:
                j_0 = mid
                r = mid - 1
            else:
                l = mid + 1
        if j_0 != -1:
            count_c += (n - j_0)
            
    result = (n - 1) * s_prime - m * count_c
    print(result)

if __name__ == '__main__':
    solve()