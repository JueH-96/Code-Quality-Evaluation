def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def check(x):
        boxes = sorted(b + [x])
        toys = sorted(a)
        
        import itertools
        for p in itertools.permutations(boxes):
            valid = True
            for i in range(n):
                if toys[i] > p[i]:
                    valid = False
                    break
            if valid:
                return True
        return False

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