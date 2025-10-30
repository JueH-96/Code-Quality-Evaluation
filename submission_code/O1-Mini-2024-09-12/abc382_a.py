N, D = map(int, input().split())
S = input()
empty_initial = S.count('.')
empty_total = empty_initial + D
print(empty_total)