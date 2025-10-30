n = int(input())
A = list(map(int, input().split()))
count = 0

while True:
    positives = sum(1 for x in A if x > 0)
    if positives <= 1:
        break
    sorted_A = sorted(A, reverse=True)
    sorted_A[0] -= 1
    sorted_A[1] -= 1
    A = sorted_A
    count += 1

print(count)