import sys
input = sys.stdin.read
data = input().split()

Q = int(data[0])
queries = data[1:]

bag = set()
results = []

i = 0
while i < len(queries):
    query_type = int(queries[i])
    if query_type == 1:
        x = int(queries[i + 1])
        bag.add(x)
        i += 2
    elif query_type == 2:
        x = int(queries[i + 1])
        bag.remove(x)
        i += 2
    elif query_type == 3:
        results.append(len(bag))
        i += 1

for result in results:
    print(result)