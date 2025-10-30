# coding: utf-8
N, M, P = map(int,input().split())
count = 0
now = M
while now <= N:
    count += 1
    now += P
print(count)