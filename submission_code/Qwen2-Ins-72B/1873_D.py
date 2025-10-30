from itertools import accumulate

for _ in range(int(input())):
    n, k = map(int, input().split())
    s = input()
    a = [0] + list(accumulate(len(x) for x in s.replace('W', ' ').split()))
    print(max((len(a) - i - 1) // k + bool(i)
              for i in range(len(a)) if a[i]))