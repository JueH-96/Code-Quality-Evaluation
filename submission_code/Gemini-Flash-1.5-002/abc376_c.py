def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def check(x):
        boxes = sorted(b + [x])
        toys = sorted(a)
        
        
        used = [False] * n
        
        for i in range(n):
            placed = False
            for j in range(n):
                if not used[j] and toys[i] <= boxes[j]:
                    used[j] = True
                    placed = True
                    break
            if not placed:
                return False
        return True

    left = 1
    right = 10**9 + 1
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