def get_divisors(n):
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i != n:
                divisors.append(i)
            if i != n//i and n//i != n:
                divisors.append(n//i)
        i += 1
    return divisors

def grundy(n):
    if n == 1:
        return 0
    divs = get_divisors(n)
    seen = set()
    for d in divs:
        seen.add(grundy(d))
    mex = 0
    while mex in seen:
        mex += 1
    return mex

N = int(input())
A = list(map(int, input().split()))

xor_sum = 0
for x in A:
    xor_sum ^= grundy(x)

if xor_sum == 0:
    print("Bruno")
else:
    print("Anna")