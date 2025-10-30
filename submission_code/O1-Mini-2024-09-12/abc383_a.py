# YOUR CODE HERE
def main():
    import sys

    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    data = N_and_rest[1:]
    times = []
    values = []
    for i in range(N):
        T = int(data[2*i])
        V = int(data[2*i+1])
        times.append(T)
        values.append(V)
    
    water = 0
    last_time = 0
    for i in range(N):
        current_time = times[i]
        V = values[i]
        delta = current_time - last_time
        water = max(water - delta, 0)
        water += V
        last_time = current_time
    print(water)

if __name__ == "__main__":
    main()