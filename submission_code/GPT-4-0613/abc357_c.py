def carpet(n):
    if n == 0:
        return ["#"]
    else:
        prev = carpet(n-1)
        top_bottom = [x*3 for x in prev]
        middle = [x + "." + x for x in prev]
        return top_bottom + middle + top_bottom

N = int(input().strip())
for line in carpet(N):
    print(line)