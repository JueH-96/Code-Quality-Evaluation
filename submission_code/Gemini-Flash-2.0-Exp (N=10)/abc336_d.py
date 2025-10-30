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

    def find_max_pyramid(arr):
        max_k = 0
        q = [arr]
        visited = {tuple(arr)}

        while q:
            curr_arr = q.pop(0)
            
            for k in range(1, (len(curr_arr) + 2) // 2):
                if is_pyramid(curr_arr, k):
                    max_k = max(max_k, k)
            
            if len(curr_arr) == 0:
                continue

            # Decrease operation
            for i in range(len(curr_arr)):
                if curr_arr[i] > 1:
                    next_arr = list(curr_arr)
                    next_arr[i] -= 1
                    next_arr_tuple = tuple(next_arr)
                    if next_arr_tuple not in visited:
                        q.append(next_arr)
                        visited.add(next_arr_tuple)

            # Remove first
            next_arr = curr_arr[1:]
            if len(next_arr) > 0:
                next_arr_tuple = tuple(next_arr)
                if next_arr_tuple not in visited:
                    q.append(next_arr)
                    visited.add(next_arr_tuple)
            
            # Remove last
            next_arr = curr_arr[:-1]
            if len(next_arr) > 0:
                next_arr_tuple = tuple(next_arr)
                if next_arr_tuple not in visited:
                    q.append(next_arr)
                    visited.add(next_arr_tuple)
        
        return max_k

    print(find_max_pyramid(a))

solve()