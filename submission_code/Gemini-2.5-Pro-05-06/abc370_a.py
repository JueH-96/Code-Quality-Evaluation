# YOUR CODE HERE
def solve():
    L, R = map(int, input().split())

    if L == 1 and R == 0:
        # Snuke is raising only his left hand.
        # This means he wants to eat takoyaki.
        print("Yes")
    elif L == 0 and R == 1:
        # Snuke is raising only his right hand.
        # This means he does not want to eat takoyaki.
        print("No")
    elif (L == 1 and R == 1) or (L == 0 and R == 0):
        # Snuke is raising both hands or no hands.
        # This is an invalid state.
        print("Invalid")
    # The constraints L, R in {0, 1} ensure these are the only possible cases.

if __name__ == "__main__":
    solve()