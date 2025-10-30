n = int(input())
a = list(map(int, input().split()))
pairs = [(a[i], i + 1) for i in range(n)]
sorted_pairs = sorted(pairs, key=lambda x: -x[0])
print(sorted_pairs[1][1])