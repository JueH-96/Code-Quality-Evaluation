# YOUR CODE HERE
N, M = map(int, input().split())

# Track which families have already had their eldest son
has_taro = [False] * (N + 1)  # Index 0 unused, families are 1-indexed

for i in range(M):
    A, B = input().split()
    A = int(A)
    
    # Check if this baby is Taro
    if B == 'M' and not has_taro[A]:
        print("Yes")
        has_taro[A] = True
    else:
        print("No")