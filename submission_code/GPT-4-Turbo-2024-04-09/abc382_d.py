def generate_sequences(N, M):
    from sys import stdout
    results = []
    
    def backtrack(current, last_value):
        if len(current) == N:
            results.append(current[:])
            return
        start = last_value + 10 if current else 1
        for value in range(start, M + 1):
            current.append(value)
            backtrack(current, value)
            current.pop()
    
    backtrack([], 0)
    
    stdout.write(f"{len(results)}
")
    for result in results:
        stdout.write(" ".join(map(str, result)) + "
")

import sys
input = sys.stdin.read
data = input().strip()
N, M = map(int, data.split())
generate_sequences(N, M)