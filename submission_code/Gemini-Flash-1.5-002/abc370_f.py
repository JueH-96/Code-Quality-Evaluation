# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

total_sum = sum(A)
left = 1
right = total_sum

def check(x):
    count = 0
    current_sum = 0
    for i in range(N):
        if current_sum + A[i] <= x:
            current_sum += A[i]
        else:
            count += 1
            current_sum = A[i]
    count += 1
    return count <= K

while left <= right:
    mid = (left + right) // 2
    if check(mid):
        left = mid + 1
    else:
        right = mid - 1

min_sum = right

ans = 0
current_sum = 0
cuts = []
for i in range(N):
    if current_sum + A[i] <= min_sum:
        current_sum += A[i]
    else:
        cuts.append(i)
        current_sum = A[i]
cuts.append(N)

count = 0
for i in range(len(cuts) -1):
    num_pieces = cuts[i+1] - cuts[i]
    if num_pieces == 1:
        count +=1

print(min_sum, count)