# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

products = []
for i in range(N):
    price = int(data[index])
    index += 1
    num_functions = int(data[index])
    index += 1
    functions = set()
    for j in range(num_functions):
        functions.add(int(data[index]))
        index += 1
    products.append((price, functions))

def is_strictly_superior(p1, p2):
    price1, functions1 = p1
    price2, functions2 = p2
    if price1 >= price2 and functions2.issuperset(functions1):
        if price1 > price2 or not functions1.issuperset(functions2):
            return True
    return False

found = False
for i in range(N):
    for j in range(N):
        if i != j and is_strictly_superior(products[i], products[j]):
            found = True
            break
    if found:
        break

if found:
    print("Yes")
else:
    print("No")