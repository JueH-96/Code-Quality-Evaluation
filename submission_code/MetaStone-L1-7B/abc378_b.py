n = int(input())
q_list = []
r_list = []
for _ in range(n):
    q, r = map(int, input().split())
    q_list.append(q)
    r_list.append(r)

q = int(input())
for _ in range(q):
    t, d = map(int, input().split())
    q_i = q_list[t-1]
    r_i = r_list[t-1]
    rem = d % q_i
    if rem == r_i:
        print(d)
    else:
        if d < r_i:
            print(r_i)
        else:
            k = (d - r_i) // q_i + 1
            print(r_i + k * q_i)