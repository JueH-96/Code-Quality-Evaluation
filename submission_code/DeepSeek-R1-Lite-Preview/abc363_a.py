R = int(input())
next_lower_bound = (R // 100 + 1) * 100
increase = next_lower_bound - R
print(increase)