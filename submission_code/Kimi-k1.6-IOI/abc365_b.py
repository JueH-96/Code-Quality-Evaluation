n = int(input())
a = list(map(int, input().split()))
elements = [(val, idx+1) for idx, val in enumerate(a)]
elements.sort(reverse=True, key=lambda x: x[0])
print(elements[1][1])