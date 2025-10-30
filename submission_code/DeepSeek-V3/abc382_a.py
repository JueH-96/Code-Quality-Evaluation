# YOUR CODE HERE
N, D = map(int, input().split())
S = input().strip()
count = S.count('@')
empty_boxes = N - (count - D)
print(empty_boxes)