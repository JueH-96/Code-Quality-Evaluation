# YOUR CODE HERE
N, Q = map(int, input().split())
treatments = list(map(int, input().split()))

teeth = [1] * N  # 1 means there is a tooth, 0 means there is no tooth

for T in treatments:
    teeth[T-1] = 1 - teeth[T-1]  # Toggle the state of the tooth

print(sum(teeth))