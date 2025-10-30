def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_sum = float('-inf')

    for i in range(1 << n):
        s = []
        ops = []
        for j in range(n):
            if (i >> j) & 1:
                ops.append(1)  # append
            else:
                ops.append(0)  # delete

        possible = True
        for j in range(n):
            if ops[j] == 1:
                s.append(a[j])
            else:
                if not s:
                    possible = False
                    break
                else:
                    s.pop()
        
        if possible:
            max_sum = max(max_sum, sum(s))
    
    print(max_sum)

solve()