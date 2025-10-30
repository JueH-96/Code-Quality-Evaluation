t = int(input().strip())
results = []

for _ in range(t):
    n = int(input().strip())
    arr = list(map(int, input().split()))
    best = -10**18
    for i in range(n):
        temp_arr = arr.copy()
        temp_arr[i] += 1
        product = 1
        for num in temp_arr:
            product *= num
        if product > best:
            best = product
    results.append(best)

for res in results:
    print(res)