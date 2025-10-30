# YOUR CODE HERE
def cycle_length(x, p):
    cycle = [x]
    while p[x-1] != cycle[0]:
        x = p[x-1]
        cycle.append(x)
    return len(cycle)

def power_mod(base, exponent, modulus):
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result

N, K = map(int, input().split())
P = list(map(int, input().split()))

result = [0] * N
for i in range(N):
    if result[i] == 0:
        cycle = []
        x = i + 1
        while result[x-1] == 0:
            cycle.append(x)
            result[x-1] = -1
            x = P[x-1]
        
        cycle_len = len(cycle)
        offset = power_mod(K, 1, cycle_len)
        
        for j, num in enumerate(cycle):
            new_pos = (j + offset) % cycle_len
            result[num-1] = cycle[new_pos]

print(' '.join(map(str, result)))