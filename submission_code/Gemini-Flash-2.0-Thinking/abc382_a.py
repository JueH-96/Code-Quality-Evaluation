N, D = map(int, input().split())
S = input()

initial_empty_count = S.count('.')
num_empty_after_days = initial_empty_count + D

print(num_empty_after_days)