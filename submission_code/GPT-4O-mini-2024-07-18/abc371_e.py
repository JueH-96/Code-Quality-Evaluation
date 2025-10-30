def distinct_subsequence_sum(N, A):
    total_sum = 0
    last_seen = {}
    left = 0
    distinct_count = 0

    for right in range(N):
        if A[right] not in last_seen:
            distinct_count += 1
        last_seen[A[right]] = right

        while left <= right:
            total_sum += distinct_count
            if left == right:
                break
            left += 1
            if last_seen[A[left - 1]] == left - 1:
                distinct_count -= 1
                del last_seen[A[left - 1]]
            else:
                last_seen[A[left - 1]] -= 1

    return total_sum

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

result = distinct_subsequence_sum(N, A)
print(result)