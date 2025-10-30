def solve():
    n, l, r = map(int, input().split())
    a = list(range(1, n + 1))

    left_index = l - 1
    right_index = r - 1

    while left_index < right_index:
        a[left_index], a[right_index] = a[right_index], a[left_index]
        left_index += 1
        right_index -= 1

    print(*a)

if __name__ == "__main__":
    solve()