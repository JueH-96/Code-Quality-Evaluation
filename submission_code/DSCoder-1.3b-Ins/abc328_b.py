# YOUR CODE HERE

N = int(input())
D = list(map(int, input().split()))

def is_repdigit(n):
    return len(set(str(n))) == 1

count = 0
for i in range(1, N+1):
    for j in range(1, D[i-1]+1):
        if is_repdigit(i) and is_repdigit(j):
            count += 1

print(count)