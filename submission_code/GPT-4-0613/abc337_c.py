N = int(input())
A = list(map(int, input().split()))
A = [a-1 for a in A]  # 0-indexed

front = A.index(-1)
order = [front]
while len(order) < N:
    for i, a in enumerate(A):
        if a == order[-1] and i not in order:
            order.append(i)
            break
print(' '.join(str(o+1) for o in order))