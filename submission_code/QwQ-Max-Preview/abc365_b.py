n = int(input())
a = list(map(int, input().split()))
elements = [(num, idx) for idx, num in enumerate(a)]
elements.sort(reverse=True, key=lambda x: x[0])
print(elements[1][1] + 1)