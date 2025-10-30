def solve():
    n, m = map(int, input().split())
    s = input()

    def is_possible(k):
        available_plain = m
        available_logo = k
        used_plain_count = 0
        used_logo_count = 0
        for day_type in s:
            if day_type == '0':
                available_plain += used_plain_count
                available_logo += used_logo_count
                used_plain_count = 0
                used_logo_count = 0
            elif day_type == '1':
                if available_plain > 0:
                    available_plain -= 1
                    used_plain_count += 1
                elif available_logo > 0:
                    available_logo -= 1
                    used_logo_count += 1
                else:
                    return False
            elif day_type == '2':
                if available_logo > 0:
                    available_logo -= 1
                    used_logo_count += 1
                else:
                    return False
        return True

    low = 0
    high = n
    ans = n
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    print(ans)

if __name__ == '__main__':
    solve()