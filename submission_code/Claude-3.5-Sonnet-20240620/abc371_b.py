# YOUR CODE HERE
N, M = map(int, input().split())

families = [False] * (N + 1)  # To keep track of families with a Taro

for _ in range(M):
    A, B = input().split()
    A = int(A)
    
    if B == 'M' and not families[A]:
        print("Yes")
        families[A] = True
    else:
        print("No")