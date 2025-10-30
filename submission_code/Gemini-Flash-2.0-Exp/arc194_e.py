def solve():
    n, x, y = map(int, input().split())
    s = list(input())
    t = list(input())

    def operate_a(arr, i):
        for j in range(y):
            arr[i+j] = '1'
        for j in range(x):
            arr[i+y+j] = '0'

    def operate_b(arr, i):
        for j in range(x):
            arr[i+j] = '0'
        for j in range(y):
            arr[i+x+j] = '1'

    def check(arr1, arr2):
        return arr1 == arr2

    q = [(s, [])]
    visited = {tuple(s)}

    while q:
        curr_s, path = q.pop(0)

        if check(curr_s, t):
            print("Yes")
            return

        for i in range(n - (x + y) + 1):
            
            # Check if Operation A is possible
            can_a = True
            for j in range(x):
                if curr_s[i+j] != '0':
                    can_a = False
                    break
            for j in range(y):
                if curr_s[i+x+j] != '1':
                    can_a = False
                    break
            
            if can_a:
                next_s = curr_s[:]
                operate_a(next_s, i)
                if tuple(next_s) not in visited:
                    q.append((next_s, path + ["A"]))
                    visited.add(tuple(next_s))

            # Check if Operation B is possible
            can_b = True
            for j in range(y):
                if curr_s[i+j] != '1':
                    can_b = False
                    break
            for j in range(x):
                if curr_s[i+y+j] != '0':
                    can_b = False
                    break
            
            if can_b:
                next_s = curr_s[:]
                operate_b(next_s, i)
                if tuple(next_s) not in visited:
                    q.append((next_s, path + ["B"]))
                    visited.add(tuple(next_s))

    print("No")

solve()