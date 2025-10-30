import sys
input = sys.stdin.read

def min_tshirts_needed(N, M, S):
    max_needed = 0
    current_needed = 0

    for i in range(N):
        if S[i] == '1':
            current_needed += 1
        elif S[i] == '2':
            current_needed += 1
        elif S[i] == '0':
            current_needed = 0

        max_needed = max(max_needed, current_needed)

    min_logo_tshirts = max(0, max_needed - M)
    return min_logo_tshirts

data = input().split()
N = int(data[0])
M = int(data[1])
S = data[2]

print(min_tshirts_needed(N, M, S))