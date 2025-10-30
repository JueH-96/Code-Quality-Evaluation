# YOUR CODE HERE

H = int(input())

def find_day(H):
    day = 0
    height = 0
    while height <= H:
        day += 1
        height = 2**day
    return day

print(find_day(H))