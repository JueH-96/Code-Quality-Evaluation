def solve():
    n = int(input())
    a = list(map(int, input().split()))

    ans = []
    for k in range(n):
        max_size = 0
        q = [(list(a), k, a[k])]
        visited = set()
        
        while q:
            curr_a, curr_k, curr_size = q.pop(0)
            
            tuple_a = tuple(curr_a)
            if (tuple_a, curr_k) in visited:
                continue
            visited.add((tuple_a, curr_k))

            max_size = max(max_size, curr_size)

            if curr_k > 0 and curr_a[curr_k - 1] < curr_size:
                next_a = curr_a[:curr_k - 1] + curr_a[curr_k:]
                next_size = curr_size + curr_a[curr_k - 1]
                q.append((next_a, curr_k - 1, next_size))
            if curr_k < len(curr_a) - 1 and curr_a[curr_k + 1] < curr_size:
                next_a = curr_a[:curr_k] + curr_a[curr_k + 2:]
                next_size = curr_size + curr_a[curr_k + 1]
                q.append((next_a, curr_k, next_size))

        ans.append(max_size)
    print(*ans)

solve()