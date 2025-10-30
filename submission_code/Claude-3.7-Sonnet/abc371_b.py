# YOUR CODE HERE
def solve():
    # Read the number of families and babies
    N, M = map(int, input().split())
    
    # Keep track of families that already have a Taro (eldest son)
    families_with_taro = set()
    
    # Process each baby in chronological order
    for _ in range(M):
        A, B = input().split()
        A = int(A)  # Family number
        
        # A baby is named Taro if:
        # 1. The baby is male (B is "M")
        # 2. The family doesn't already have a Taro (first male child)
        if B == "M" and A not in families_with_taro:
            print("Yes")
            families_with_taro.add(A)  # Mark that this family now has a Taro
        else:
            print("No")

solve()