input()
lst = list(map(int, input().split()))
sMax = sorted(lst)[-2]
print(lst.index(sMax) + 1)