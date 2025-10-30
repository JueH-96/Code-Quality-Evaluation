import sys

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    L = []
    R = []
    for _ in range(M):
        li = int(input[ptr])
        ptr += 1
        ri = int(input[ptr])
        ptr += 1
        L.append(li)
        R.append(ri)

    used = [False] * M
    ans = [0] * M
    current = 1
    cost = 0

    while current <= N:
        best_reach = -1
        best_i = -1
        best_type = -1

        for i in range(M):
            if used[i]:
                continue
            Li = L[i]
            Ri = R[i]

            # Check operation 1
            if Li <= current <= Ri:
                reach = Ri
                if reach > best_reach:
                    best_reach = reach
                    best_i = i
                    best_type = 1

            # Check operation 2's left part (current < Li)
            if current < Li:
                reach = Li - 1
                if reach > best_reach:
                    best_reach = reach
                    best_i = i
                    best_type = 2

            # Check operation 2's right part (current > Ri)
            if current > Ri:
                reach = N
                if reach > best_reach:
                    best_reach = reach
                    best_i = i
                    best_type = 2

        if best_i == -1:
            print(-1)
            return

        used[best_i] = True
        ans[best_i] = best_type
        cost += 1
        current = best_reach + 1

    print(cost)
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()