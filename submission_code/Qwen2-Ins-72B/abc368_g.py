from collections import deque

def process_queries(N, A, B, Q, queries):
    ans = []
    q = deque()
    for i in range(N):
        q.append((A[i], B[i]))
    
    for i in range(Q):
        query = queries[i]
        if query[0] == 1:
            q[query[1]-1] = (query[2], q[query[1]-1][1])
        elif query[0] == 2:
            q[query[1]-1] = (q[query[1]-1][0], query[2])
        else:
            l, r = query[1]-1, query[2]
            plus, mul = 0, 1
            while l <= r:
                if l == r:
                    plus += q[l][0]
                    mul *= q[l][1]
                    break
                plus += q[l][0]
                mul *= q[r][1]
                l += 1
                r -= 1
            ans.append(max(plus, mul))
    return ans

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
Q = int(input())
queries = [list(map(int, input().split())) for _ in range(Q)]

for i in process_queries(N, A, B, Q, queries):
    print(i)