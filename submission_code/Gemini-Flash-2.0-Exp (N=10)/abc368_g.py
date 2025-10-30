def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        query = list(map(int, input().split()))
        type = query[0]

        if type == 1:
            i, x = query[1], query[2]
            a[i - 1] = x
        elif type == 2:
            i, x = query[1], query[2]
            b[i - 1] = x
        else:
            l, r = query[1], query[2]
            
            def calculate_max_value(index, current_value, current_path):
                if index == r:
                    return current_value
                
                add_value = calculate_max_value(index + 1, current_value + a[index], current_path + [0])
                mult_value = calculate_max_value(index + 1, current_value * b[index], current_path + [1])
                
                return max(add_value, mult_value)

            print(calculate_max_value(l - 1, 0, []))
            

solve()