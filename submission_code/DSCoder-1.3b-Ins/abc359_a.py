# YOUR CODE HERE
N = int(input())
S = input().split()
count = sum([1 for s in S if s == 'Takahashi'])
print(count)