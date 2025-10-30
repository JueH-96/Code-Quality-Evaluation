import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
q = int(data[1])
s = list(data[2])
queries = data[3:]

def is_good_string(substring):
    for i in range(1, len(substring)):
        if substring[i] == substring[i - 1]:
            return "No"
    return "Yes"

results = []
for query in queries:
    query_type, l, r = int(query[0]), int(query[2]) - 1, int(query[4]) - 1
    if query_type == 1:
        for i in range(l, r + 1):
            s[i] = '0' if s[i] == '1' else '1'
    elif query_type == 2:
        results.append(is_good_string(s[l:r + 1]))

for result in results:
    print(result)