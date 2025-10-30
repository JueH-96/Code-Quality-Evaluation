from collections import Counter

Q = int(input().strip())
counter = Counter()

for _ in range(Q):
    query = input().strip().split()
    if query[0] == '1':
        counter[query[1]] += 1
    elif query[0] == '2':
        counter[query[1]] -= 1
        if counter[query[1]] == 0:
            del counter[query[1]]
    else:
        print(len(counter))