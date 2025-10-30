# YOUR CODE HERE
N = int(input())
a = N // 100
b = (N // 10) % 10
c = N % 10
print(f"{b}{c}{a} {c}{a}{b}")