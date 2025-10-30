# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = int(data[1])
    T = list(map(int, data[2:2+N]))
    
    finish_time = 0
    
    for i in range(N):
        arrival_time = T[i]
        if arrival_time >= finish_time:
            finish_time = arrival_time + A
        else:
            finish_time += A
        print(finish_time)

if __name__ == "__main__":
    main()