import bisect

n = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A_sorted = sorted(A)
B_sorted = sorted(B)

low = 1
high = 10**18
answer = -1

while low <= high:
    mid = (low + high) // 2
    k = bisect.bisect_right(B_sorted, mid)
    valid = True
    for i in range(n):
        if i < k:
            if B_sorted[i] < A_sorted[i]:
                valid = False
                break
        elif i == k:
            if mid < A_sorted[i]:
                valid = False
                break
        else:
            if B_sorted[i-1] < A_sorted[i]:
                valid = False
                break
    if valid:
        answer = mid
        high = mid - 1
    else:
        low = mid + 1

print(answer if answer != -1 else -1)