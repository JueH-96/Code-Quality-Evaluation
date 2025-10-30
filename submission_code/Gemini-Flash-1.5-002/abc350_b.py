# YOUR CODE HERE
n, q = map(int, input().split())
treatments = list(map(int, input().split()))

teeth = [True] * n

for treatment in treatments:
    if teeth[treatment - 1]:
        teeth[treatment - 1] = False
    else:
        teeth[treatment - 1] = True

print(sum(teeth))