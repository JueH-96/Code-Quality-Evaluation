# YOUR CODE HERE
def determine_hand_signal():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    L, R = map(int, data.split())
    
    if L == 1 and R == 0:
        print("Yes")
    elif L == 0 and R == 1:
        print("No")
    else:
        print("Invalid")

determine_hand_signal()