def solve():
    n = int(input())
    a = list(map(int, input().split()))

    results = []
    for k in range(n):
        max_size = 0
        for start_dir in [0, 1]:  # 0: left, 1: right
            current_size = a[k]
            current_a = list(a)
            current_k = k

            while True:
                found_smaller = False
                if start_dir == 0:
                    dirs = [0, 1]
                else:
                    dirs = [1, 0]

                for direction in dirs:
                    if direction == 0: #left
                        if current_k > 0 and current_a[current_k - 1] < current_size:
                            current_size += current_a[current_k - 1]
                            current_a.pop(current_k - 1)
                            current_k -= 1
                            found_smaller = True
                            break
                    else: #right
                        if current_k < len(current_a) - 1 and current_a[current_k + 1] < current_size:
                            current_size += current_a[current_k + 1]
                            current_a.pop(current_k + 1)
                            found_smaller = True
                            break
                
                if not found_smaller:
                    break
            max_size = max(max_size, current_size)
        results.append(max_size)

    print(*results)

solve()