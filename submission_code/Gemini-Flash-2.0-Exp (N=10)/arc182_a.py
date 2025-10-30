def solve():
    n, q = map(int, input().split())
    ops = []
    for _ in range(q):
        ops.append(list(map(int, input().split())))

    count = 0
    
    def check_valid(s, p, v, op_type):
        if op_type == 0:
            for i in range(p):
                if s[i] > v:
                    return False
        else:
            for i in range(p - 1, n):
                if s[i] > v:
                    return False
        return True

    def generate_sequences(index, current_s, current_ops):
        nonlocal count
        if index == q:
            count = (count + 1) % 998244353
            return

        p, v = ops[index]
        
        # Option 1: Replace S[0] to S[P-1]
        temp_s1 = list(current_s)
        if check_valid(temp_s1, p, v, 0):
            for i in range(p):
                temp_s1[i] = v
            generate_sequences(index + 1, temp_s1, current_ops + [0])
        
        # Option 2: Replace S[P-1] to S[N-1]
        temp_s2 = list(current_s)
        if check_valid(temp_s2, p, v, 1):
            for i in range(p - 1, n):
                temp_s2[i] = v
            generate_sequences(index + 1, temp_s2, current_ops + [1])

    initial_s = [0] * n
    generate_sequences(0, initial_s, [])
    print(count)

solve()