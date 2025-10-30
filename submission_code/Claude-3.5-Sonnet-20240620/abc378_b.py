# YOUR CODE HERE
def next_collection_day(q, r, d):
    return d + (q - d % q + r) % q

N = int(input())
garbage_types = []
for _ in range(N):
    q, r = map(int, input().split())
    garbage_types.append((q, r))

Q = int(input())
for _ in range(Q):
    t, d = map(int, input().split())
    q, r = garbage_types[t - 1]
    print(next_collection_day(q, r, d))