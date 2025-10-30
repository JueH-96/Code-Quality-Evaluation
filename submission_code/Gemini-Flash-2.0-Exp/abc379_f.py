def solve():
    n, q = map(int, input().split())
    h = list(map(int, input().split()))
    
    def can_see(i, j, heights):
        for k in range(i + 1, j):
            if heights[k] > heights[j]:
                return False
        return True
    
    for _ in range(q):
        l, r = map(int, input().split())
        l -= 1
        r -= 1
        
        count = 0
        for j in range(r + 1, n):
            if can_see(l, j, h) and can_see(r, j, h):
                count += 1
        print(count)

solve()