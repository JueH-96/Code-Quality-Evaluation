A, M, L, R = map(int, input().split())

if L > R:
    L, R = R, L

start = (L - A) // M
if (L - A) % M < 0:
    start -= 1

end = (R - A) // M

print(end - start + 1)