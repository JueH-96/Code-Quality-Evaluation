# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T_prime = data[1]
S = data[2:N+2]

def is_valid(T, T_prime):
    if T == T_prime:
        return True
    if abs(len(T) - len(T_prime)) > 1:
        return False
    if len(T) == len(T_prime):
        diff = 0
        for a, b in zip(T, T_prime):
            if a != b:
                diff += 1
        return diff == 1
    if len(T) < len(T_prime):
        T, T_prime = T_prime, T
    for i in range(len(T)):
        if T[:i] + T[i+1:] == T_prime:
            return True
    return False

result = []
for i, s in enumerate(S):
    if is_valid(s, T_prime):
        result.append(i + 1)

print(len(result))
print(*result)