N, K = map(int, input().split())
A = list(map(int, input().split()))

counter = 0
empty = K

while A:
    if A[0] > empty:
        counter += 1
        empty = K
    else:
        empty -= A[0]
        A.pop(0)

# Start the attraction one last time after all groups are processed
counter += 1

print(counter)