def solve():
    n, q = map(int, input().split())
    p = []
    v = []
    for _ in range(q):
        pi, vi = map(int, input().split())
        p.append(pi)
        v.append(vi)

    count = 0
    
    def check_crying(s, pi, vi, choice):
        if choice == 0:
            for i in range(pi):
                if s[i] > vi:
                    return True
        else:
            for i in range(pi - 1, n):
                if s[i] > vi:
                    return True
        return False

    def perform_operation(s, pi, vi, choice):
        new_s = s[:]
        if choice == 0:
            for i in range(pi):
                new_s[i] = vi
        else:
            for i in range(pi - 1, n):
                new_s[i] = vi
        return new_s

    def find_valid_sequences(index, current_s):
        nonlocal count
        if index == q:
            count = (count + 1) % 998244353
            return

        pi = p[index]
        vi = v[index]

        # Choice 0
        if not check_crying(current_s, pi, vi, 0):
            new_s = perform_operation(current_s, pi, vi, 0)
            find_valid_sequences(index + 1, new_s)

        # Choice 1
        if not check_crying(current_s, pi, vi, 1):
            new_s = perform_operation(current_s, pi, vi, 1)
            find_valid_sequences(index + 1, new_s)

    initial_s = [0] * n
    find_valid_sequences(0, initial_s)
    print(count)

solve()