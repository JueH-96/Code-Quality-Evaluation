import math

def count_good_integers(N):
    count = 0
    a = 1
    while (1 << a) <= N:  # 2^a <= N
        max_b = int(math.sqrt(N // (1 << a)))
        count += (max_b + 1) // 2  # Count of odd integers from 1 to max_b
        a += 1
    return count

N = int(input())
print(count_good_integers(N))