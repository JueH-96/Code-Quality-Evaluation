def solve():
    n, m = map(int, input().split())
    s = input()

    def can_satisfy(num_logo_shirts_to_buy):
        plain_shirts_available = m
        logo_shirts_available = num_logo_shirts_to_buy
        plain_shirts_worn = 0
        logo_shirts_worn = 0

        for day_schedule in s:
            if day_schedule == '0':
                plain_shirts_available += plain_shirts_worn
                logo_shirts_available += logo_shirts_worn
                plain_shirts_worn = 0
                logo_shirts_worn = 0
            elif day_schedule == '1':
                if plain_shirts_available > 0:
                    plain_shirts_available -= 1
                    plain_shirts_worn += 1
                elif logo_shirts_available > 0:
                    logo_shirts_available -= 1
                    logo_shirts_worn += 1
                else:
                    return False
            elif day_schedule == '2':
                if logo_shirts_available > 0:
                    logo_shirts_available -= 1
                    logo_shirts_worn += 1
                else:
                    return False
        return True

    low = 0
    high = n
    ans = n

    while low <= high:
        mid = (low + high) // 2
        if can_satisfy(mid):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    print(ans)

if __name__ == '__main__':
    solve()