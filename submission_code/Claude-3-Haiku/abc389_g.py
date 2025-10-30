import math

def count_graphs(N, P):
    results = []
    for M in range(N-1, (N*(N-1))//2 + 1):
        count = 0
        for i in range(1 << (N-1)):
            if bin(i).count('1') == M:
                even_count = 0
                for j in range(1, N):
                    if (i >> (j-1)) & 1:
                        if (j-1) % 2 == 0:
                            even_count += 1
                if even_count == N - even_count:
                    count += 1
        results.append(count % P)
    return ' '.join(map(str, results))

# Read input from stdin
N, P = map(int, input().split())

# Solve the problem and write output to stdout
print(count_graphs(N, P))