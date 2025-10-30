def solve():
    n = int(input())
    a = list(map(int, input().split()))

    min_passengers = 0
    current_passengers = 0
    
    for change in a:
        current_passengers += change
        min_passengers = min(min_passengers, current_passengers)
    
    print(max(0, current_passengers - min_passengers))

solve()