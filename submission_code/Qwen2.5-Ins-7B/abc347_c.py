# N A B
# D_1 D_2 ... D_N
N, A, B = map(int, input().split())
D = list(map(int, input().split()))

# Check if all plans can be on holidays
# If the difference between any two consecutive plans is less than or equal to A, it's impossible
for i in range(1, N):
    if D[i] - D[i-1] <= A:
        print('No')
        exit()

print('Yes')