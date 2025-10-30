n = int(input())
a = list(map(int, input().split()))
elements = [(a[i], i + 1) for i in range(n)]
elements.sort(key=lambda x: (-x[0], x[1]))
print(elements[1][1])