# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    C = int(data[1])
    T = list(map(int, data[2:]))
    
    candies = 1  # Always get a candy on the first press
    last_candy_time = T[0]
    
    for i in range(1, N):
        if T[i] - last_candy_time >= C:
            candies += 1
            last_candy_time = T[i]
    
    print(candies)

if __name__ == "__main__":
    main()