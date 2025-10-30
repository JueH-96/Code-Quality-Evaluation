def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def is_pyramid(arr, k):
        if len(arr) != 2 * k - 1:
            return False
        for i in range(k):
            if arr[i] != i + 1:
                return False
        for i in range(k, 2 * k - 1):
            if arr[i] != 2 * k - 1 - i:
                return False
        return True

    def check(k, arr):
        if 2 * k - 1 > len(arr):
            return False

        q = [(arr, [])]
        visited = {tuple(arr)}

        while q:
            curr_arr, ops = q.pop(0)

            if len(curr_arr) == 2 * k - 1:
                if is_pyramid(curr_arr, k):
                    return True

            if len(curr_arr) > 2 * k - 1:
                # Remove first
                next_arr1 = curr_arr[1:]
                if tuple(next_arr1) not in visited:
                    q.append((next_arr1, ops + ["remove_first"]))
                    visited.add(tuple(next_arr1))

                # Remove last
                next_arr2 = curr_arr[:-1]
                if tuple(next_arr2) not in visited:
                    q.append((next_arr2, ops + ["remove_last"]))
                    visited.add(tuple(next_arr2))
            elif len(curr_arr) < 2 * k - 1:
                continue
            else:
                if is_pyramid(curr_arr, k):
                    return True
                else:
                    return False

            # Decrease
            for i in range(len(curr_arr)):
                if curr_arr[i] > 1:
                    next_arr = curr_arr[:]
                    next_arr[i] -= 1
                    if tuple(next_arr) not in visited:
                        q.append((next_arr, ops + ["decrease"]))
                        visited.add(tuple(next_arr))

        return False

    ans = 0
    for k in range(1, n // 2 + 2):
        if check(k, a):
            ans = k
        else:
            break
    print(ans)

solve()