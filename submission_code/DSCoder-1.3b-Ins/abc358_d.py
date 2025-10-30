N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Check if it is possible to buy M boxes that can satisfy the condition
if max(A) > sum(B):
    print(-1)
else:
    # Calculate the minimum total amount of money Takahashi needs to pay
    total_price = sum(A) + sum(B)
    print(total_price)