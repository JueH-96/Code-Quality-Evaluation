# coding: utf-8
# Your code here!
import sys
read = sys.stdin.read
readline = sys.stdin.readline
readlines = sys.stdin.readlines

n, r = map(int,readline().split())

for i in range(n):
    d,a = map(int,readline().split())
    if (d==1 and 1600<=r<=2799) or (d==2 and 1200<=r<=2399):
        r += a

print(r)