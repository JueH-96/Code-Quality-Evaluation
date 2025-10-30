# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    medicines = []
    for _ in range(n):
        medicines.append(list(map(int, input().split())))

    def check(day):
        total_pills = 0
        for a, b in medicines:
            if day <= a:
                total_pills += b
        return total_pills <= k

    left = 1
    right = 2 * 10**9
    ans = -1

    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            ans = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(ans)


solve()