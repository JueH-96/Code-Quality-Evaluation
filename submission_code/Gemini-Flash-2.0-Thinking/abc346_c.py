def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    total_sum = k * (k + 1) // 2

    present_numbers = set()
    for x in a:
        if 1 <= x <= k:
            present_numbers.add(x)

    sum_of_present = sum(present_numbers)

    missing_sum = total_sum - sum_of_present
    print(missing_sum)

solve()