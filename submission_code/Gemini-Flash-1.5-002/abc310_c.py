# YOUR CODE HERE
def solve():
    n = int(input())
    sticks = []
    for _ in range(n):
        sticks.append(input())
    
    distinct_sticks = set()
    for stick in sticks:
        reversed_stick = stick[::-1]
        if stick in distinct_sticks or reversed_stick in distinct_sticks:
            continue
        else:
            distinct_sticks.add(stick)

    print(len(distinct_sticks))

solve()