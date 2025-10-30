# YOUR CODE HERE
def solve(N, K, A):
    prefix_sum = [0] * (2*N + 1)
    for i in range(1, 2*N + 1):
        prefix_sum[i] = prefix_sum[i-1] + A[(i-1) % N]

    def check(x):
        count = 0
        last = 0
        for i in range(1, N + 1):
            if prefix_sum[i] - prefix_sum[last] >= x:
                count += 1
                last = i
        return count >= K

    left, right = 0, sum(A)
    while right - left > 1:
        mid = (left + right) // 2
        if check(mid):
            left = mid
        else:
            right = mid
    
    min_sum = left

    not_cut = 0
    last = 0
    for i in range(1, N + 1):
        if prefix_sum[i] - prefix_sum[last] >= min_sum:
            not_cut += i - last - 1
            last = i
    not_cut += N - last

    return min_sum, not_cut

N, K = map(int, input().split())
A = list(map(int, input().split()))

min_sum, not_cut = solve(N, K, A)
print(min_sum, not_cut)