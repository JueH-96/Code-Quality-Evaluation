# YOUR CODE HERE
def find_culprit_oyster():
    import sys
    input = sys.stdin.read
    S_1, S_2 = input().strip().split()

    if S_1 == "sick" and S_2 == "fine":
        print(2)
    elif S_1 == "fine" and S_2 == "sick":
        print(3)
    elif S_1 == "sick" and S_2 == "sick":
        print(1)
    elif S_1 == "fine" and S_2 == "fine":
        print(4)