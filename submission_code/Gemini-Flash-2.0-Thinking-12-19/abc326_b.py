def is_326_like(num):
    s_num = str(num)
    h = int(s_num[0])
    t = int(s_num[1])
    o = int(s_num[2])
    return h * t == o

n = int(input())
current_num = n
while True:
    if is_326_like(current_num):
        print(current_num)
        break
    current_num += 1