def solve():
    n = int(input())
    sticks = []
    for _ in range(n):
        sticks.append(input())
    
    unique_sticks = set()
    for stick in sticks:
        reversed_stick = stick[::-1]
        if stick in unique_sticks or reversed_stick in unique_sticks:
            continue
        else:
            unique_sticks.add(stick)
    
    print(len(unique_sticks))

solve()