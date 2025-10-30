def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    result = [0] * N
    j = M - 1  # Start from the last fireworks day
    
    for i in range(N, 0, -1):
        if j >= 0 and i == A[j]:
            result[i-1] = 0
            j -= 1
        else:
            if j >= 0:
                result[i-1] = A[j] - i
            else:
                result[i-1] = A[0] - i  # In case j goes below 0, though it shouldn't happen as A includes day N
    
    print('
'.join(map(str, result)))

if __name__ == '__main__':
    main()