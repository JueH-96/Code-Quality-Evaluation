# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    C = int(data[1])
    T = list(map(int, data[2:]))
    
    candies = 0
    last_candy_time = -C  # Initialize to a time that guarantees the first candy is given
    
    for t in T:
        if t - last_candy_time >= C:
            candies += 1
            last_candy_time = t
    
    print(candies)

if __name__ == "__main__":
    main()