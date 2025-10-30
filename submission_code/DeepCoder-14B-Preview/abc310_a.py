n, p, q = map(int, input().split())
dishes = list(map(int, input().split()))
min_dish = min(dishes)
option1 = p
option2 = q + min_dish
print(min(option1, option2))