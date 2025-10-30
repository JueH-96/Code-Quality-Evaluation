import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    Q = int(data[idx])
    idx += 1
    
    current_time = 0
    created_times = []
    start = 0
    
    for _ in range(Q):
        if data[idx] == '1':
            created_times.append(current_time)
            idx +=1
        elif data[idx] == '2':
            T = int(data[idx+1])
            current_time += T
            idx +=2
        else:
            H = int(data[idx+1])
            X = current_time - H
            idx +=2
            pos = bisect.bisect_right(created_times, X, start)
            print(pos - start)
            start = pos

if __name__ == "__main__":
    main()