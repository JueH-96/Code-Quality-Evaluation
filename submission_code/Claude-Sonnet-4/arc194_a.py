n = int(input())
a = list(map(int, input().split()))

stack = []

for i in range(n):
    if not stack:
        stack.append(a[i])
    else:
        current_sum = sum(stack)
        sum_if_append = current_sum + a[i]
        sum_if_delete = current_sum - stack[-1]
        
        if sum_if_append >= sum_if_delete:
            stack.append(a[i])
        else:
            stack.pop()

print(sum(stack))