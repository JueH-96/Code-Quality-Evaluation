import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
A = list(map(int, data[3:]))

remaining_votes = K - sum(A)
sorted_A = sorted(A, reverse=True)

def can_win(i, additional_votes):
    votes_needed = A[i] + additional_votes
    count_greater = 0
    for votes in sorted_A:
        if votes > votes_needed:
            count_greater += 1
    return count_greater < M

result = []

for i in range(N):
    if can_win(i, 0):
        result.append(0)
    else:
        low, high = 0, remaining_votes
        min_votes = -1
        while low <= high:
            mid = (low + high) // 2
            if can_win(i, mid):
                min_votes = mid
                high = mid - 1
            else:
                low = mid + 1
        result.append(min_votes if min_votes != -1 else -1)

print(" ".join(map(str, result)))