import bisect

N, K = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

B_sorted = sorted(B)

def count_ge(V):
    count = 0
    for i in range(N):
        for k in range(N):
            S = A[i] + C[k]
            T = V - C[k] * A[i]
            
            if S > 0:
                if T <= 0:
                    count += N
                else:
                    threshold = (T + S - 1) // S  # ceiling division
                    idx = bisect.bisect_left(B_sorted, threshold)
                    count += N - idx
            elif S < 0:
                if T >= 0:
                    count += 0
                else:
                    threshold = T // S  # floor division
                    idx = bisect.bisect_right(B_sorted, threshold)
                    count += idx
            else:  # S == 0
                if T <= 0:
                    count += N
    
    return count

min_val = min(A) * min(B) + min(B) * min(C) + min(C) * min(A)
max_val = max(A) * max(B) + max(B) * max(C) + max(C) * max(A)

left, right = min_val, max_val
while left < right:
    mid = (left + right + 1) // 2
    if count_ge(mid) >= K:
        left = mid
    else:
        right = mid - 1

print(left)