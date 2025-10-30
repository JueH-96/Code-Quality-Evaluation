# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    
    # Read the 3D array A
    A = [[[0] * (N + 1) for _ in range(N + 1)] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                A[x][y][z] = int(data[index])
                index += 1
    
    Q = int(data[index])
    index += 1
    
    results = []
    
    for _ in range(Q):
        Lx = int(data[index])
        Rx = int(data[index + 1])
        Ly = int(data[index + 2])
        Ry = int(data[index + 3])
        Lz = int(data[index + 4])
        Rz = int(data[index + 5])
        index += 6
        
        total_sum = 0
        
        for x in range(Lx, Rx + 1):
            for y in range(Ly, Ry + 1):
                for z in range(Lz, Rz + 1):
                    total_sum += A[x][y][z]
        
        results.append(total_sum)
    
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()