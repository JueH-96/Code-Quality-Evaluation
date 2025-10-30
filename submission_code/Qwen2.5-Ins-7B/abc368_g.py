# YOUR CODE HERE
from sys import stdin, stdout

class QueryProcessor:
    def __init__(self, N, A, B):
        self.N = N
        self.A = A
        self.B = B
        self.A_updates = [0] * (N + 1)
        self.B_updates = [0] * (N + 1)
    
    def process_query(self, query):
        t, i, x = query
        if t == 1:
            self.A_updates[i] = x
        elif t == 2:
            self.B_updates[i] = x
        else:
            return self.solve_query(i, x)
    
    def solve_query(self, l, r):
        max_v = 0
        A = self.A[:l] + [0] * (r - l + 1) + self.A[r+1:]
        B = self.B[:l] + [0] * (r - l + 1) + self.B[r+1:]
        for i in range(l, r + 1):
            A[i] += self.A_updates[i]
            B[i] += self.B_updates[i]
        for i in range(l, r + 1):
            if A[i] > 0:
                max_v = max(max_v, max_v * B[i])
            else:
                max_v += A[i]
        return max_v

N = int(stdin.readline())
A = list(map(int, stdin.readline().split()))
B = list(map(int, stdin.readline().split()))
Q = int(stdin.readline())
queries = [list(map(int, stdin.readline().split())) for _ in range(Q)]

processor = QueryProcessor(N, A, B)
for query in queries:
    stdout.write(f"{processor.process_query(query)}
")