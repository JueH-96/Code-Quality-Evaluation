# YOUR CODE HERE
N, D = map(int, input().split())
S = input()
num_empty_boxes = S.count('.') + D
print(num_empty_boxes)