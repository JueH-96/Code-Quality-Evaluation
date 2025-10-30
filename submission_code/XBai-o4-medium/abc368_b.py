n = int(input())
A = list(map(int, input().split()))
count = 0

while True:
    pos_count = sum(1 for x in A if x > 0)
    if pos_count <= 1:
        break
    # Sort in descending order
    A.sort(reverse=True)
    # Decrease the first two elements
    A[0] -= 1
    A[1] -= 1
    count += 1

print(count)