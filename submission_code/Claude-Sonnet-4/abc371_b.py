# YOUR CODE HERE
N, M = map(int, input().split())

# Track which families have already had their eldest son (first male child)
has_eldest_son = [False] * (N + 1)  # Index 0 unused, families are 1-indexed

for i in range(M):
    family, gender = input().split()
    family = int(family)
    
    # Check if this baby gets named Taro
    if gender == 'M' and not has_eldest_son[family]:
        # This is the first male child in this family
        print("Yes")
        has_eldest_son[family] = True
    else:
        # Either not male, or not the first male in this family
        print("No")