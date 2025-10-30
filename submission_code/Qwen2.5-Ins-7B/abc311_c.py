# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

for i in range(n):
    if a[a[i]-1] == i+1:
        cycle = [i+1, a[i]]
        j = a[i]-1
        while a[j]-1 != i:
            cycle.append(a[j])
            j = a[j]-1
        break

print(len(cycle))
print(*cycle)