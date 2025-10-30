# YOUR CODE HERE
import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
dependencies = defaultdict(list)
index = 1

for i in range(1, N + 1):
    C_i = int(data[index])
    dependencies[i] = list(map(int, data[index + 1:index + 1 + C_i]))
    index += 1 + C_i

# To find the order of books to read to read book 1
def find_order_to_read(book, dependencies, visited, stack):
    visited[book] = True
    for dep in dependencies[book]:
        if not visited[dep]:
            find_order_to_read(dep, dependencies, visited, stack)
    stack.append(book)

visited = [False] * (N + 1)
stack = []

find_order_to_read(1, dependencies, visited, stack)

# We need to print the books excluding book 1
stack.pop()  # Remove book 1
stack.reverse()
print(" ".join(map(str, stack)))