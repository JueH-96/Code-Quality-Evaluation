def solve():
    n, q = map(int, input().split())
    requirements = list(map(int, input().split()))
    sorted_requirements = sorted(requirements)
    prefix_sums = [0] * (n + 1)
    for i in range(n):
        prefix_sums[i+1] = prefix_sums[i] + sorted_requirements[i]
    
    results = []
    for _ in range(q):
        x = int(input())
        low = 0
        high = n
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            current_sum = prefix_sums[mid]
            if current_sum <= x:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        results.append(ans)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()