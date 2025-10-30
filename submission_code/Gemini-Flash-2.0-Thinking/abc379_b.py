def solve():
    n, k = map(int, input().split())
    s = list(input())
    strawberries = 0

    while True:
        found_index = -1
        for i in range(n - k + 1):
            is_healthy = True
            for j in range(i, i + k):
                if s[j] == 'X':
                    is_healthy = False
                    break
            if is_healthy:
                found_index = i
                break

        if found_index != -1:
            strawberries += 1
            for j in range(found_index, found_index + k):
                s[j] = 'X'
        else:
            break
    print(strawberries)

solve()