# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    C = int(data[1])
    T = list(map(int, data[2:]))
    
    candies = 0
    last_candy_time = -float('inf')
    
    for i in range(N):
        if T[i] - last_candy_time >= C:
            candies += 1
            last_candy_time = T[i]
    
    print(candies)

if __name__ == "__main__":
    main()