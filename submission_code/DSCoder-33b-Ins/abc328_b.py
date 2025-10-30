# YOUR CODE HERE

N = int(input())
D = list(map(int, input().split()))

def is_repdigit(i, j):
    return len(set(str(i))) == 1 and len(set(str(j))) == 1

repdigit_days = 0
for i in range(N):
    for j in range(1, D[i]+1):
        if is_repdigit(i+1, j):
            repdigit_days += 1

print(repdigit_days)