def is_326_like(num):
    s_num = str(num)
    if len(s_num) != 3:
        return False
    return int(s_num[0]) * int(s_num[1]) == int(s_num[2])

n = int(input())
while True:
    if is_326_like(n):
        print(n)
        break
    n += 1