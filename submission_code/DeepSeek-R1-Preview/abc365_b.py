n = int(input())
a = list(map(int, input().split()))
indexed = [(value, idx + 1) for idx, value in enumerate(a)]
sorted_list = sorted(indexed, key=lambda x: x[0], reverse=True)
print(sorted_list[1][1])