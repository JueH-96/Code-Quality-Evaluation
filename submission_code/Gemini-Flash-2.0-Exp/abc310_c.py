def solve():
    n = int(input())
    sticks = []
    for _ in range(n):
        sticks.append(input())
    
    unique_sticks = set()
    for stick in sticks:
        reversed_stick = stick[::-1]
        
        found = False
        for unique_stick in unique_sticks:
            if stick == unique_stick or reversed_stick == unique_stick:
                found = True
                break
        
        if not found:
            unique_sticks.add(stick)
    
    print(len(unique_sticks))

solve()