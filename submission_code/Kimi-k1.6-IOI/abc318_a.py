n, m, p = map(int, input().split())
print(0 if m > n else (n - m) // p + 1)