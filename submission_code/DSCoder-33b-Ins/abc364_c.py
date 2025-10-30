# YOUR CODE HERE

n, x, y = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

dishes = sorted([(a[i], b[i]) for i in range(n)], key=lambda x: x[0] + x[1], reverse=True)

sweetness = saltiness = count = 0
for dish in dishes:
    if sweetness + dish[0] <= x and saltiness + dish[1] <= y:
        sweetness += dish[0]
        saltiness += dish[1]
        count += 1

print(count)