import bisect

def lis_length_with_max(arr, end, max_val):
    dp = []
    for i in range(end):
        num = arr[i]
        if num <= max_val:
            pos = bisect.bisect_left(dp, num)
            if pos == len(dp):
                dp.append(num)
            else:
                dp[pos] = num
    return len(dp)

# Read input
n, q = map(int, input().split())
a = list(map(int, input().split()))

for _ in range(q):
    r, x = map(int, input().split())
    result = lis_length_with_max(a, r, x)
    print(result)