# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

Q = int(data[0])
queries = data[1:]

bag = set()
results = []

for i in range(Q):
    query = queries[i]
    if query[0] == '1':
        x = int(query[2:])
        bag.add(x)
    elif query[0] == '2':
        x = int(query[2:])
        bag.remove(x)
    elif query[0] == '3':
        results.append(len(bag))

for result in results:
    print(result)