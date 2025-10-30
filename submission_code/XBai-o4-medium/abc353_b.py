# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

current_empty = K
count = 0
i = 0

while i < N:
    if A[i] <= current_empty:
        current_empty -= A[i]
        i += 1
    else:
        count += 1
        current_empty = K

count += 1  # Start the attraction once when the queue is empty

print(count)