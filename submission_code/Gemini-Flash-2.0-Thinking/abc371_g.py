def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))

    p = [x - 1 for x in p]

    def apply_operation(arr):
        next_arr = [0] * n
        for i in range(n):
            next_arr[i] = arr[p[i]]
        return next_arr

    def is_lex_smaller(arr1, arr2):
        for i in range(n):
            if arr1[i] < arr2[i]:
                return True
            if arr1[i] > arr2[i]:
                return False
        return False

    smallest_a = list(a)
    seen_a = {tuple(a)}
    current_a = list(a)

    while True:
        next_a = apply_operation(current_a)

        if is_lex_smaller(next_a, smallest_a):
            smallest_a = list(next_a)

        if tuple(next_a) in seen_a:
            break

        seen_a.add(tuple(next_a))
        current_a = next_a

    print(*smallest_a)

solve()