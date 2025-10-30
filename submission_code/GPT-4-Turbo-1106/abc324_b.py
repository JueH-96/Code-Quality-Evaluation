def check_power_of_2_and_3(N):
    while N % 2 == 0:
        N //= 2
    while N % 3 == 0:
        N //= 3
    return N == 1

N = int(input())
print("Yes" if check_power_of_2_and_3(N) else "No")