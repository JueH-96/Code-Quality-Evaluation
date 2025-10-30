def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def count_divisors(num):
        count = 0
        for i in range(1, int(num**0.5) + 1):
            if num % i == 0:
                count += 1
                if i * i != num:
                    count += 1
        return count - 1

    total_moves = 0
    for num in a:
        total_moves += count_divisors(num)

    if total_moves % 2 == 1:
        print("Anna")
    else:
        print("Bruno")

solve()