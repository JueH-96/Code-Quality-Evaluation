def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            i, x = query[1], query[2]
            a[i-1] = x
        elif query[0] == 2:
            i, x = query[1], query[2]
            b[i-1] = x
        else:
            l, r = query[1], query[2]
            
            def calculate_max(index, current_value):
                if index == r:
                    return max(current_value + a[index-1], current_value * b[index-1])
                else:
                    return max(calculate_max(index+1, current_value + a[index-1]), calculate_max(index+1, current_value * b[index-1]))

            print(calculate_max(l, 0))

solve()