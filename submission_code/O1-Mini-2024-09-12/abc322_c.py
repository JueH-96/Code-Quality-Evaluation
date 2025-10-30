# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    result = [0] * N
    j = M -1
    next_firework_day = N  # Last day is always a fireworks day
    
    for i in range(N, 0, -1):
        if j >=0 and A[j] == i:
            next_firework_day = i
            j -=1
        result[i-1] = next_firework_day - i
    
    print('
'.join(map(str, result)))

if __name__ == "__main__":
    main()