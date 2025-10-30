# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    M = int(input[0])
    S1 = input[1]
    S2 = input[2]
    S3 = input[3]
    
    min_time = float('inf')
    found = False
    
    for target in range(10):
        times = []
        for i in range(M):
            if S1[i] == str(target):
                times.append(i)
            if S2[i] == str(target):
                times.append(i)
            if S3[i] == str(target):
                times.append(i)
        
        if len(times) < 3:
            continue
        
        times.sort()
        for i in range(len(times) - 2):
            for j in range(i + 1, len(times) - 1):
                for k in range(j + 1, len(times)):
                    max_time = max(times[i], times[j], times[k])
                    if max_time < min_time:
                        min_time = max_time
                        found = True
    
    if found:
        print(min_time)
    else:
        print(-1)

if __name__ == "__main__":
    main()