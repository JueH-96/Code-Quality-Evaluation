def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    C = int(input_data[1])
    T = list(map(int, input_data[2:2+N]))
    
    candies = 0
    last_received = -C  # This initialization guarantees the first press gives a candy.
    
    for t in T:
        if t - last_received >= C:
            candies += 1
            last_received = t
            
    print(candies)

main()