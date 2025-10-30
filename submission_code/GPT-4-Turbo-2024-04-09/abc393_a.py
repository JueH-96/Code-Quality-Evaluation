# YOUR CODE HERE
def find_problematic_oyster():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    S_1, S_2 = data[0], data[1]
    
    # Takahashi ate oysters 1 and 2
    # Aoki ate oysters 1 and 3
    
    # If Takahashi is sick and Aoki is fine, the problematic oyster must be 2
    if S_1 == "sick" and S_2 == "fine":
        print(2)
    # If Takahashi is fine and Aoki is sick, the problematic oyster must be 3
    elif S_1 == "fine" and S_2 == "sick":
        print(3)
    # If both are sick, the problematic oyster must be 1
    elif S_1 == "sick" and S_2 == "sick":
        print(1)
    # If both are fine, the problematic oyster must be 4
    elif S_1 == "fine" and S_2 == "fine":
        print(4)