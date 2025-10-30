def solve():
    n = int(input())
    x = list(map(int, input().split()))

    def calculate_sum(arr):
        return sum(arr)

    def perform_operation(arr, i):
        arr.sort()
        m = (arr[i] + arr[i+3]) / 2
        arr[i+1] = m + (m - arr[i+1])
        arr[i+2] = m - (arr[i+2] - m)
        return sorted(arr)

    q = [sorted(x)]
    visited = {tuple(sorted(x))}
    min_sum = calculate_sum(x)

    while q:
        curr_x = q.pop(0)
        min_sum = min(min_sum, calculate_sum(curr_x))

        for i in range(n - 3):
            next_x = perform_operation(curr_x[:], i)
            if tuple(next_x) not in visited:
                visited.add(tuple(next_x))
                q.append(next_x)
    
    print(int(min_sum))

solve()