# YOUR CODE HERE
def solve():
    n = int(input())
    sticks = []
    for _ in range(n):
        sticks.append(input())
    
    unique_sticks = set()
    for stick in sticks:
        if stick not in unique_sticks and stick[::-1] not in unique_sticks:
            unique_sticks.add(stick)
            
    print(len(unique_sticks))

solve()