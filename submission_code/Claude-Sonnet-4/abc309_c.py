# YOUR CODE HERE
def solve():
    N, K = map(int, input().split())
    medicines = []
    for _ in range(N):
        a, b = map(int, input().split())
        medicines.append((a, b))
    
    # Binary search on the day
    left, right = 1, max(a for a, b in medicines) + 1
    
    def pills_on_day(day):
        total = 0
        for a, b in medicines:
            if day <= a:
                total += b
        return total
    
    # Find the first day where pills <= K
    while left < right:
        mid = (left + right) // 2
        if pills_on_day(mid) <= K:
            right = mid
        else:
            left = mid + 1
    
    print(left)

solve()