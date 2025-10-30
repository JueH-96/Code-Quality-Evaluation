# YOUR CODE HERE
R = int(input())
next_hundred = ((R // 100) + 1) * 100  # Next 'hundredth' mark
increase = next_hundred - R
print(increase)