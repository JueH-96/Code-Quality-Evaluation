import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = int(data[1])
    T = list(map(int, data[2:]))
    
    finish_times = []
    current_time = 0
    
    for i in range(N):
        if i == 0:
            current_time = T[i] + A
        else:
            if T[i] >= current_time:
                current_time = T[i] + A
            else:
                current_time += A
        finish_times.append(current_time)
    
    for time in finish_times:
        print(time)

if __name__ == "__main__":
    main()