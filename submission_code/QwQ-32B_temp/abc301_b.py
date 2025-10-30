n = int(input())
A = list(map(int, input().split()))

while True:
    all_ok = True
    for i in range(len(A) - 1):
        if abs(A[i] - A[i+1]) != 1:
            all_ok = False
            break
    if all_ok:
        break
    # Find the first problematic pair
    first_i = 0
    for i in range(len(A) - 1):
        if abs(A[i] - A[i+1]) != 1:
            first_i = i
            break
    a, b = A[first_i], A[first_i + 1]
    if a < b:
        insert_list = list(range(a + 1, b))
    else:
        insert_list = list(range(a - 1, b, -1))
    # Update the array
    A = A[:first_i + 1] + insert_list + A[first_i + 1:]

print(' '.join(map(str, A)))