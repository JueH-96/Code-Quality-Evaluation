import sys

N, T_prime = input().split()
N = int(N)
T_prime = T_prime.strip()

S_list = []
for _ in range(N):
    S_list.append(input().strip())

def is_possible(T, T_prime):
    if T == T_prime:
        return True
    if len(T) == len(T_prime) - 1:
        for i in range(len(T) + 1):
            if T[:i] + T[i:] == T_prime:
                return True
    if len(T) == len(T_prime) + 1:
        for i in range(len(T)):
            if T[:i] + T[i+1:] == T_prime:
                return True
    if len(T) == len(T_prime):
        for i in range(len(T)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if T[:i] + c + T[i+1:] == T_prime:
                    return True
    return False

result = []
for i, S in enumerate(S_list):
    if is_possible(S, T_prime):
        result.append(i + 1)

print(len(result))
print(' '.join(map(str, result)))