n = int(input())
A = list(map(int, input().split()))

while True:
    all_consec = True
    first_i = -1
    for i in range(len(A) - 1):
        if abs(A[i] - A[i+1]) != 1:
            all_consec = False
            first_i = i
            break
    if all_consec:
        break
    a = A[first_i]
    b = A[first_i + 1]
    if a < b:
        insert_nums = list(range(a + 1, b))
    else:
        insert_nums = list(range(a - 1, b, -1))
    A = A[:first_i + 1] + insert_nums + A[first_i + 1:]

print(' '.join(map(str, A)))