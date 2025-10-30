# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = []
    V = []
    for i in range(N):
        T.append(int(data[2*i+1]))
        V.append(int(data[2*i+2]))
    
    current_water = 0
    prev_time = 0
    
    for i in range(N):
        current_time = T[i]
        time_diff = current_time - prev_time
        current_water = max(0, current_water - time_diff)
        current_water += V[i]
        prev_time = current_time
    
    print(current_water)

if __name__ == "__main__":
    main()