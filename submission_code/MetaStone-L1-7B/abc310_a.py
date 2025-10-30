n, p, q = map(int, input().split())
d_list = list(map(int, input().split()))
min_d = min(d_list)
option1 = q + min_d
option2 = p
print(min(option1, option2))