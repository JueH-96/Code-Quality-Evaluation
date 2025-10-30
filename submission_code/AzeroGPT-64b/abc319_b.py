n = int(input())
ans = ['-']* (n+1)

def fun(j, l):
    for i in range(l, n+1,l):
        if ans[i] == '-':
            ans[i] = str(j)
for i in range(1,10):
    if n%i == 0:
        fun(i, n//i)
print(''.join(ans))