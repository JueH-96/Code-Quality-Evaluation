# YOUR CODE HERE
import sys

def find_middle_occurrences(A, N):
    f = {}
    for i in range(1, N + 1):
        f[i] = []
    
    for index, value in enumerate(A, start=1):
        f[value].append(index)
    
    sorted_indices = sorted(f.keys(), key=lambda x: f[x][1])
    return sorted_indices

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = find_middle_occurrences(A, N)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()