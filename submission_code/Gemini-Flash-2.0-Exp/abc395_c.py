def solve():
    n = int(input())
    a = list(map(int, input().split()))

    min_len = float('inf')

    for i in range(n):
        for j in range(i, n):
            sub_array = a[i:j+1]
            if len(sub_array) > 0:
                counts = {}
                for x in sub_array:
                    if x in counts:
                        counts[x] += 1
                    else:
                        counts[x] = 1
                
                repeated = False
                for x in counts:
                    if counts[x] > 1:
                        repeated = True
                        break
                
                if repeated:
                    min_len = min(min_len, len(sub_array))
    
    if min_len == float('inf'):
        print("-1")
    else:
        print(min_len)

solve()